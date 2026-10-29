"""Same-cohort LR/RF controls, trained and selected before common final evaluation."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from common import HERE,digest,log,save
from cost_models import class_costs,probability
from experiment import source_data,evaluation_data,benchmark
from operating_points import source_point,select_cost
from protocol import metrics,summaries

PARAMETERS=dict(lr=dict(C=1.,max_iter=2000),rf=dict(n_estimators=200,n_jobs=4))


def train():
    config=json.loads((HERE/'config.json').read_text());files=[]
    if (HERE/'results/evaluation_start_common.json').exists():
        raise RuntimeError('These baseline controls must be frozen before common final evaluation.')
    for seed in config['seeds']:
        for source in config['primary_pair']:
            directory=HERE/'models/common_baselines'/source/str(seed);directory.mkdir(parents=True,exist_ok=True)
            main=HERE/'models/common'/source/str(seed);info=json.loads((main/'frozen.json').read_text())
            x,y,xv,yv,audit=source_data('common',source,seed)
            if audit!=info['source_audit']:
                raise AssertionError('Baseline source cohort differs from teacher/student cohort.')
            transform=joblib.load(main/'transform.joblib');xt=transform.apply(x);xc=transform.apply(xv)
            candidates=[];artifacts={}
            for family in PARAMETERS:
                for cost in config['benign_cost_grid']:
                    model=LogisticRegression(random_state=seed,**PARAMETERS[family]) if family=='lr' else RandomForestClassifier(random_state=seed,**PARAMETERS[family])
                    weights=class_costs(y,cost);model.fit(xt,y,sample_weight=weights[y])
                    if family=='rf':
                        model.set_params(n_jobs=1)
                    filename=family+'_cost'+str(cost)+'.joblib';joblib.dump(model,directory/filename)
                    artifacts[filename]=digest(directory/filename);scores=probability(model,xc)
                    for alpha in config['source_fpr_targets']:
                        candidates.append(dict(family=family,candidate_id=filename,benign_cost=cost,**source_point(yv,scores,alpha)))
                    log(f'BASELINE TRAIN PASS {source}/seed{seed}/{family}/cost{cost}: same source train/calibration records')
            chosen=[]
            for family in PARAMETERS:
                for alpha in config['source_fpr_targets']:
                    options=[{k:v for k,v in r.items() if k!='family'} for r in candidates if r['family']==family and r['fpr_limit']==alpha]
                    standard=next(r for r in options if r['benign_cost']==1)
                    for arm,choice in [('standard',standard),('cost_sensitive',select_cost(options))]:
                        chosen.append(dict(family=family,arm=arm,**choice))
            file=directory/'frozen.json';save(file,dict(source=source,seed=seed,source_audit=audit,
                main_source_freeze_sha256=digest(main/'frozen.json'),transform_sha256=digest(main/'transform.joblib'),
                candidates=candidates,selections=chosen,artifact_sha256=artifacts,
                parameters=PARAMETERS,external_metrics_used_for_selection=False,code_sha256=digest(__file__)))
            files.append(file)
    save(HERE/'manifests/frozen_common_baselines.json',dict(files={str(f.relative_to(HERE)):digest(f) for f in files},
        code_sha256=digest(__file__),config_sha256=digest(HERE/'config.json'),final_performance_seen=False))
    log('BASELINE FREEZE COMPLETE: 100 LR/RF fits; all choices source-only and capture/record matched')


def evaluate():
    frozen=HERE/'manifests/frozen_common_baselines.json';manifest=json.loads(frozen.read_text())
    if manifest['code_sha256']!=digest(__file__):
        raise RuntimeError('Baseline code changed after source freeze.')
    output=[];cpu=[]
    for path,sha in manifest['files'].items():
        file=HERE/path
        if digest(file)!=sha:
            raise AssertionError('Baseline choice record changed.')
        info=json.loads(file.read_text());source=info['source'];seed=info['seed'];directory=file.parent
        main=HERE/'models/common'/source/str(seed)
        if digest(main/'transform.joblib')!=info['transform_sha256'] or digest(main/'frozen.json')!=info['main_source_freeze_sha256']:
            raise AssertionError('Shared source data transform/choice record changed.')
        transform=joblib.load(main/'transform.joblib');cache={};benchmarked=set()
        for target in ['cic_iot_2023','ton_iot']:
            frame,y,_=evaluation_data('common',target,seed,source==target);x=transform.apply(frame);scores={}
            for choice in info['selections']:
                name=choice['candidate_id']
                if digest(directory/name)!=info['artifact_sha256'][name]:
                    raise AssertionError('Baseline artifact changed.')
                if name not in cache:
                    cache[name]=joblib.load(directory/name)
                model=cache[name]
                if name not in scores:
                    scores[name]=probability(model,x)
                p=scores[name];values=metrics(y,(p>=choice['threshold']).astype(int),p,p)
                output.append(dict(source=source,target=target,seed=seed,family=choice['family'],arm=choice['arm'],
                    benign_cost=choice['benign_cost'],source_fpr_limit=choice['fpr_limit'],threshold=choice['threshold'],
                    calibration_recall=choice['source_recall'],calibration_fpr=choice['source_fpr'],
                    evaluation='within' if source==target else 'external',**values))
                if name not in benchmarked:
                    cpu.append(dict(source=source,seed=seed,family=choice['family'],benign_cost=choice['benign_cost'],
                        **benchmark(model,transform,frame)));benchmarked.add(name)
            log(f'BASELINE EVALUATION PASS {source}->{target}/seed{seed}: frozen LR/RF policies')
    data=pd.DataFrame(output)
    if len(data)!=160 or (data.calibration_fpr>data.source_fpr_limit+1e-12).any():
        raise AssertionError('Baseline results incomplete or source FPR violation.')
    data.to_csv(HERE/'results/common_baseline_performance.csv',index=False)
    summaries(data,['source','target','family','arm','source_fpr_limit','evaluation'],['macro_f1','recall','fpr','balanced_accuracy']).to_csv(
        HERE/'results/common_baseline_summary.csv',index=False)
    pd.DataFrame(cpu).to_csv(HERE/'results/common_baseline_cpu.csv',index=False)
    save(HERE/'results/common_baseline_audit.json',dict(rows=len(data),freeze_sha256=digest(frozen),
        same_source_records_and_transform=True,all_sources_frozen_before_common_final=True,
        all_model_and_freeze_hashes_verified=True,not_independent_extra_test=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['train','evaluate'],required=True);a=p.parse_args()
    train() if a.stage=='train' else evaluate()
