from __future__ import annotations

import argparse
import copy
import io
import json
import os
import platform
import sys
import time
import warnings
from pathlib import Path

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
import joblib
import numpy as np
import pandas as pd
from scipy.special import expit
from scipy.stats import ks_2samp, spearmanr
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import torch
import psutil
from protocol import (group_split, hashes, metrics, source_threshold, select_on_source,
    alarm_rule, conservative_alarm_predict, summaries)
from data_revision import ROOT, load_pool, prepare, write_manifest
from neural_revision import train_neural, scores as neural_scores

sys.path.insert(0, str(ROOT/'src'))
from litexids.data.domain_adaptation import fit_coral, apply_coral

OUT = ROOT/'revision'/'results'
CONFIG = json.loads((ROOT/'revision'/'config.json').read_text())
DATASETS = ['cic_iot_2023', 'ton_iot', 'cic_iomt_2024']
VALUES = ['accuracy', 'macro_f1', 'balanced_accuracy', 'precision', 'recall',
    'specificity', 'fpr', 'pr_auc', 'roc_auc', 'false_alarms_per_10000', 'alarm_rate']


def write(rows, filename):
    pd.DataFrame(rows).to_csv(OUT/filename, index=False)


def partition(data, seed):
    remaining, test = group_split(data['x'], data['y'], .2, seed)
    a, b = group_split(data['x'].iloc[remaining], data['y'][remaining], .15/.8, seed+1)
    return remaining[a], remaining[b], test


class Transform:
    def __init__(self, variant='baseline'):
        self.variant = variant

    def fit(self, source, target=None):
        self.columns = list(source.columns)
        self.imputer = SimpleImputer(strategy='median', keep_empty_features=True)
        s = self.imputer.fit_transform(source)
        t = self.imputer.transform(target) if target is not None else None
        self.robust = self.variant not in ['baseline', 'iw']
        if self.robust:
            s = np.sign(s)*np.log1p(np.abs(s))
            if t is not None:
                t = np.sign(t)*np.log1p(np.abs(t))
            reference = np.vstack([s, t]) if t is not None else s
            self.lo, self.hi = np.quantile(reference, [.005, .995], axis=0)
            s = np.clip(s, self.lo, self.hi)
            t = np.clip(t, self.lo, self.hi) if t is not None else None
        self.ks = np.array([ks_2samp(s[:, j], t[:, j]).statistic for j in range(s.shape[1])]) if t is not None else np.zeros(s.shape[1])
        self.selected = np.argsort(self.ks, kind='stable')[:8] if 'gate' in self.variant else np.arange(s.shape[1])
        self.scaler = StandardScaler().fit(s[:, self.selected])
        s = self.scaler.transform(s[:, self.selected])
        t = self.scaler.transform(t[:, self.selected]) if t is not None else None
        self.coral = fit_coral(s, t) if 'coral' in self.variant and t is not None else None
        return self

    def apply(self, frame, source=False):
        x = self.imputer.transform(frame[self.columns])
        if self.robust:
            x = np.clip(np.sign(x)*np.log1p(np.abs(x)), self.lo, self.hi)
        x = self.scaler.transform(x[:, self.selected])
        if self.coral is not None and source:
            x = apply_coral(x, self.coral)
        return x


def build_model(kind, seed):
    if kind == 'lr':
        return LogisticRegression(max_iter=2000, class_weight='balanced', random_state=seed)
    if kind == 'rf':
        return RandomForestClassifier(n_estimators=200, n_jobs=4,
            class_weight='balanced_subsample', random_state=seed)
    if kind == 'xgb':
        return XGBClassifier(n_estimators=300, max_depth=6, learning_rate=.05,
            subsample=.9, colsample_bytree=.9, n_jobs=4, eval_metric='logloss', random_state=seed)
    if kind == 'lgbm':
        return LGBMClassifier(n_estimators=300, learning_rate=.05, class_weight='balanced',
            verbosity=-1, n_jobs=4, random_state=seed)
    raise ValueError(kind)


def get_scores(model, x):
    if isinstance(model, torch.nn.Module):
        z = neural_scores(model, x)
        return z, expit(z), 0.
    if hasattr(model, 'decision_function'):
        z = model.decision_function(x)
        return z, expit(z), 0.
    p = model.predict_proba(x)[:, 1]
    return p, p, .5


def fit(kind, train, y, cal, yc, seed, target=None, weights=None):
    if kind in ['mlp', 'tcn', 'dann']:
        return train_neural(kind, train, y, cal, yc, seed,
            epochs=CONFIG['neural_epochs'], target=target)
    model = build_model(kind, seed)
    model.fit(train, y, sample_weight=weights)
    return model


def importance_weights(source, target, seed):
    domain = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=seed)
    domain.fit(np.vstack([source, target]), np.r_[np.zeros(len(source)), np.ones(len(target))])
    probability = domain.predict_proba(source)[:, 1]
    w = np.clip(probability/np.clip(1-probability, 1e-6, 1), .05, 20)
    return w/w.mean()


def benchmark(model, x):
    sample = np.asarray(x[:1], dtype=np.float32)
    original_jobs = model.get_params().get('n_jobs') if hasattr(model, 'get_params') else None
    if hasattr(model, 'get_params') and 'n_jobs' in model.get_params():
        model.set_params(n_jobs=1)
    torch.set_num_threads(1)
    for _ in range(CONFIG['benchmark_warmup']):
        get_scores(model, sample)
    process = psutil.Process()
    before = process.memory_info().rss
    elapsed = []
    for _ in range(CONFIG['benchmark_repetitions']):
        start = time.perf_counter_ns()
        get_scores(model, sample)
        elapsed.append((time.perf_counter_ns()-start)/1e6)
    after = process.memory_info().rss
    stream = io.BytesIO()
    if isinstance(model, torch.nn.Module):
        torch.save(model.state_dict(), stream)
        params = sum(p.numel() for p in model.parameters())
    else:
        joblib.dump(model, stream, compress=0)
        params = np.nan
    torch.set_num_threads(4)
    if original_jobs is not None:
        model.set_params(n_jobs=original_jobs)
    return dict(model_bytes=stream.tell(), latency_mean_ms=float(np.mean(elapsed)),
        latency_p50_ms=float(np.quantile(elapsed, .5)), latency_p95_ms=float(np.quantile(elapsed, .95)),
        latency_p99_ms=float(np.quantile(elapsed, .99)), rss_bytes=after,
        rss_change_bytes=after-before, parameters=params, batch_size=1,
        warmup=CONFIG['benchmark_warmup'], repetitions=CONFIG['benchmark_repetitions'], inference_threads=1,
        throughput_per_second=1000/float(np.mean(elapsed)), scope='model and score extraction only; preprocessing excluded')


def within(pools, seeds, n):
    rows, audits, splits = [], [], []
    for seed in seeds:
        for name in DATASETS:
            d = prepare(name, pools[name], n, seed)
            audits.append(d['audit'])
            train, cal, test = partition(d, seed)
            for label, idx in [('train', train), ('calibration', cal), ('test', test)]:
                splits.append(dict(dataset=name, seed=seed, split=label, n=len(idx),
                    benign=int((d['y'][idx] == 0).sum()), attack=int(d['y'][idx].sum())))
            for space in ['native_numeric', 'proxy12', 'proxy8']:
                x = d['native'] if space == 'native_numeric' else d['x']
                if space == 'proxy8':
                    x = x.drop(columns=['bytes_min', 'bytes_max', 'rate', 'iat_proxy'])
                transform = Transform('baseline').fit(x.iloc[train])
                a, b, c = [transform.apply(x.iloc[idx], source=True) for idx in [train, cal, test]]
                for kind in ['lr', 'rf', 'xgb', 'lgbm']:
                    model = fit(kind, a, d['y'][train], b, d['y'][cal], seed)
                    score_cal, _, _ = get_scores(model, b)
                    cutoff = source_threshold(d['y'][cal], score_cal)
                    s, p, _ = get_scores(model, c)
                    rows.append(dict(dataset=name, seed=seed, space=space, model=kind,
                        threshold=cutoff, features=x.shape[1], train_n=len(a), test_n=len(c),
                        **metrics(d['y'][test], (s >= cutoff).astype(int), s, p)))
                write(rows, 'within_runs.csv')
            print(f'within {name} seed={seed}', flush=True)
    write(audits, 'projection_audit.csv')
    write(splits, 'within_splits.csv')
    summaries(pd.DataFrame(rows), ['dataset', 'space', 'model'], VALUES).to_csv(OUT/'within_summary.csv', index=False)


def explain(model, transform, train, cal, target, base, rows):
    import shap
    rng = np.random.default_rng(base['seed'])
    background = train[rng.choice(len(train), min(100, len(train)), replace=False)]
    samples = {'source': cal[rng.choice(len(cal), min(100, len(cal)), replace=False)],
        'target': target[rng.choice(len(target), min(100, len(target)), replace=False)]}
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, background)
    elif isinstance(model, (RandomForestClassifier, XGBClassifier, LGBMClassifier)):
        explainer = shap.TreeExplainer(model)
    else:
        return
    for domain, x in samples.items():
        values = np.asarray(explainer.shap_values(x))
        if values.ndim == 3:
            values = values[:, :, 1]
        importance = np.abs(values).mean(axis=0)
        for col, value in zip(np.asarray(transform.columns)[transform.selected], importance):
            rows.append(dict(**base, domain=domain, feature=col, mean_abs_shap=float(value)))


def cross(pools, seeds, n):
    rows, selected_rows, split_rows, drift_rows, shap_rows, score_rows, family_rows = [], [], [], [], [], [], []
    for seed in seeds:
        datasets = {name: prepare(name, pools[name], n, seed) for name in DATASETS}
        for source_name in DATASETS:
            source = datasets[source_name]
            train_idx, cal_idx, source_test_idx = partition(source, seed)
            for target_name in DATASETS:
                if source_name == target_name:
                    continue
                target = datasets[target_name]
                adapt_idx, eval_idx = group_split(target['x'], target['y'], .5, seed+2, require_both=False)
                for label, idx in [('adaptation', adapt_idx), ('evaluation', eval_idx)]:
                    split_rows.append(dict(source=source_name, target=target_name, seed=seed, split=label,
                        n=len(idx), benign=int((target['y'][idx] == 0).sum()), attack=int(target['y'][idx].sum())))
                cache, candidates = {}, []
                variants = ['baseline', 'robust', 'iw', 'robust_iw', 'robust_coral', 'robust_gate', 'robust_gate_iw']
                for variant in variants:
                    transform = Transform(variant).fit(source['x'].iloc[train_idx], target['x'].iloc[adapt_idx])
                    for col, ks in zip(transform.columns, transform.ks):
                        drift_rows.append(dict(source=source_name, target=target_name, seed=seed,
                            variant=variant, feature=col, ks=float(ks)))
                    a, b = [transform.apply(source['x'].iloc[idx], source=True) for idx in [train_idx, cal_idx]]
                    u, e = [transform.apply(target['x'].iloc[idx]) for idx in [adapt_idx, eval_idx]]
                    kinds = ['lr', 'rf', 'xgb', 'lgbm', 'mlp', 'tcn', 'dann'] if variant == 'baseline' else ['lr']
                    for kind in kinds:
                        w = importance_weights(a, u, seed) if variant.endswith('iw') or variant == 'iw' else None
                        model = fit(kind, a, source['y'][train_idx], b, source['y'][cal_idx], seed,
                            target=u if kind == 'dann' else None, weights=w)
                        s_cal, _, default = get_scores(model, b)
                        cutoff = source_threshold(source['y'][cal_idx], s_cal)
                        m_cal = metrics(source['y'][cal_idx], (s_cal >= cutoff).astype(int), s_cal)
                        key = f'{kind}/{variant}'
                        candidates.append(dict(id=key, source_f1=m_cal['macro_f1'], source_ba=m_cal['balanced_accuracy'],
                            complexity=0 if kind == 'lr' else 1))
                        cache[key] = (model, transform, a, b, u, e, s_cal, cutoff, default)
                # Selection occurs before target evaluation labels enter any metric calculation.
                winner = select_on_source(candidates)
                for key, item in cache.items():
                    model, transform, a, b, u, e, s_cal, cutoff, default = item
                    kind, variant = key.split('/')
                    s_adapt, _, _ = get_scores(model, u)
                    s_eval, p_eval, _ = get_scores(model, e)
                    base = dict(source=source_name, target=target_name, seed=seed, model=kind,
                        variant=variant, selected=(key == winner), source_train_n=len(a),
                        target_eval_n=len(e), features=a.shape[1])
                    rules = [('default', default, None), ('source_f1', cutoff, None),
                        ('source_fpr01', source_threshold(source['y'][cal_idx], s_cal, .01), None),
                        ('source_fpr05', source_threshold(source['y'][cal_idx], s_cal, .05), None)]
                    for budget in CONFIG['alarm_budgets']:
                        rule = alarm_rule(s_adapt, budget)
                        rules += [(f'budget_{budget:g}', rule.threshold, rule)]
                    for mode, threshold, rule in rules:
                        if rule is not None:
                            pred = rule.predict(s_eval, target['uid'][eval_idx], seed)
                            adapt_rate = rule.predict(s_adapt, target['uid'][adapt_idx], seed).mean()
                            conservative_rate = conservative_alarm_predict(s_adapt, rule).mean()
                        else:
                            pred = (s_eval >= threshold).astype(int)
                            adapt_rate = (s_adapt >= threshold).mean()
                            conservative_rate = np.nan
                        result = dict(**base, operating_point=mode, threshold=float(threshold),
                            boundary_tie_probability=rule.tie_probability if rule else np.nan,
                            realized_adaptation_alarm_rate=float(adapt_rate),
                            conservative_adaptation_alarm_rate=conservative_rate,
                            **metrics(target['y'][eval_idx], pred, s_eval, p_eval))
                        rows.append(result)
                        if key == winner:
                            selected_rows.append(result)
                        if mode == 'source_f1' and (key == winner or key == 'lr/baseline'):
                            source_families = set(source['family'][train_idx])
                            for fam in np.unique(target['family'][eval_idx]):
                                mask = target['family'][eval_idx] == fam
                                if fam == 'benign':
                                    continue
                                family_rows.append(dict(**base, family=fam, shared=fam in source_families,
                                    n=int(mask.sum()), attack_recall=float(pred[mask].mean())))
                    if key == winner or key == 'lr/baseline':
                        explain(model, transform, a, b, e, base, shap_rows)
                    if key == 'lr/baseline':
                        for domain, values in [('adaptation', s_adapt), ('evaluation', s_eval)]:
                            for q in [0, .001, .01, .05, .1, .2, .5, .8, .9, .95, .99, 1]:
                                score_rows.append(dict(**base, domain=domain, quantile=q,
                                    raw_margin=float(np.quantile(values, q)), probability=float(expit(np.quantile(values, q))),
                                    saturated_fraction=float(np.mean((expit(values) == 0)|(expit(values) == 1)))))
                    if seed == seeds[0] and key == 'rf/baseline' and source_name == 'ton_iot' and target_name == 'cic_iot_2023':
                        joblib.dump(dict(model=model, x=e[:100], y=target['y'][eval_idx][:100]), OUT/'edge_reference.joblib')
                for records, filename in [(rows,'cross_runs.csv'), (selected_rows,'selected_cross_runs.csv'),
                    (shap_rows,'external_shap.csv'), (family_rows,'attack_family_runs.csv'),
                    (score_rows,'score_distributions.csv'), (split_rows,'cross_splits.csv'), (drift_rows,'drift_runs.csv')]:
                    write(records, filename)
                print(f'cross {source_name} -> {target_name} seed={seed} winner={winner}', flush=True)
    frame = pd.DataFrame(rows)
    summaries(frame, ['source','target','model','variant','operating_point'], VALUES).to_csv(OUT/'cross_summary.csv', index=False)
    summaries(pd.DataFrame(selected_rows), ['source','target','operating_point'], VALUES).to_csv(OUT/'selected_cross_summary.csv', index=False)


def compression(pools, seeds, n):
    from torch.nn.utils import prune
    rows, cost_rows = [], []
    for seed in seeds:
        for name in DATASETS:
            d = prepare(name, pools[name], n, seed)
            train, cal, test = partition(d, seed)
            transform = Transform('robust').fit(d['x'].iloc[train])
            a,b,c = [transform.apply(d['x'].iloc[idx], source=True) for idx in [train,cal,test]]
            teacher = fit('xgb', a, d['y'][train], b, d['y'][cal], seed)
            teacher_proba = teacher.predict_proba(a)
            variants = {'teacher_xgb':teacher}
            supervised = train_neural('mlp', a, d['y'][train], b, d['y'][cal], seed,
                epochs=CONFIG['neural_epochs'])
            distilled = train_neural('mlp', a, d['y'][train], b, d['y'][cal], seed,
                epochs=CONFIG['neural_epochs'], teacher_proba=teacher_proba,
                alpha=CONFIG['distillation_alpha_hard'], temperature=CONFIG['distillation_temperature'])
            variants.update(student_supervised=supervised, student_distilled=distilled)
            pruned = copy.deepcopy(distilled)
            for layer in pruned.modules():
                if isinstance(layer, torch.nn.Linear):
                    prune.l1_unstructured(layer, name='weight', amount=.2)
                    prune.remove(layer, 'weight')
            variants['student_pruned'] = pruned
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                variants['student_pruned_quantized'] = torch.ao.quantization.quantize_dynamic(
                    copy.deepcopy(pruned), {torch.nn.Linear}, dtype=torch.qint8)
            for variant, model in variants.items():
                s_cal,_,_ = get_scores(model,b)
                cutoff = source_threshold(d['y'][cal],s_cal)
                for target_name in DATASETS:
                    td = prepare(target_name,pools[target_name],n,seed)
                    idx = test if target_name == name else group_split(td['x'],td['y'],.5,seed+2,require_both=False)[1]
                    e = c if target_name == name else transform.apply(td['x'].iloc[idx])
                    s,p,_ = get_scores(model,e)
                    rows.append(dict(source=name,target=target_name,seed=seed,variant=variant,
                        epochs=getattr(model,'trained_epochs',np.nan),temperature=CONFIG['distillation_temperature'],
                        alpha_hard=CONFIG['distillation_alpha_hard'],
                        **metrics(td['y'][idx],(s>=cutoff).astype(int),s,p)))
                cost_rows.append(dict(dataset=name,seed=seed,variant=variant,**benchmark(model,c)))
                if name == 'ton_iot' and seed == seeds[0] and variant == 'student_distilled':
                    torch.save(dict(state_dict=model.state_dict(), input_features=a.shape[1]),OUT/'student_for_edge.pt')
                    np.savez_compressed(OUT/'edge_input.npz',x=c.astype(np.float32),y=d['y'][test])
                    teacher.save_model(OUT/'teacher_for_edge.ubj')
            write(rows,'compression_runs.csv')
            write(cost_rows,'cpu_benchmark_runs.csv')
            print(f'compression {name} seed={seed}',flush=True)
    summaries(pd.DataFrame(rows),['source','target','variant'],VALUES).to_csv(OUT/'compression_summary.csv',index=False)
    summaries(pd.DataFrame(cost_rows),['dataset','variant'],['model_bytes','latency_mean_ms','latency_p95_ms','rss_bytes']).to_csv(OUT/'cpu_benchmark_summary.csv',index=False)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--stage',choices=['within','cross','compression','all'],default='all')
    parser.add_argument('--seeds',default=','.join(map(str,CONFIG['seeds'])))
    parser.add_argument('--rows',type=int,default=CONFIG['sample_rows_per_dataset'])
    args=parser.parse_args()
    seeds=list(map(int,args.seeds.split(',')))
    pools={name:load_pool(name) for name in DATASETS}
    write_manifest()
    import sklearn, scipy, shap, xgboost, lightgbm
    environment=dict(python=sys.version,platform=platform.platform(),processor=platform.processor(),
        physical_cores=psutil.cpu_count(logical=False),logical_cores=psutil.cpu_count(),
        ram_bytes=psutil.virtual_memory().total,cpu_threads=4,
        cpu_model=os.environ.get('PROCESSOR_IDENTIFIER'),
        packages={m.__name__:m.__version__ for m in [np,pd,sklearn,scipy,shap,xgboost,lightgbm,torch]},
        config=CONFIG,command=sys.argv)
    (OUT/f'environment_{args.stage}.json').write_text(json.dumps(environment,indent=2),encoding='utf-8')
    if args.stage in ['within','all']:
        within(pools,seeds,args.rows)
    if args.stage in ['cross','all']:
        cross(pools,seeds,args.rows)
    if args.stage in ['compression','all']:
        compression(pools,seeds,args.rows)


if __name__=='__main__':
    main()
