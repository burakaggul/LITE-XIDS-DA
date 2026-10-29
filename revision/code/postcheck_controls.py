"""Additional controls and a documented native-representation correction."""
from __future__ import annotations
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import warnings
import numpy as np
import pandas as pd
from data_revision import ROOT,load_pool,prepare
from protocol import hashes,group_split,source_threshold,metrics,summaries
from run_revision import CONFIG,VALUES,Transform,fit,get_scores,benchmark,partition

OUT=ROOT/'revision'/'results'


def fingerprint():
    pool=load_pool('ton_iot')
    result={}
    for seed in CONFIG['seeds']:
        d=prepare('ton_iot',pool,CONFIG['sample_rows_per_dataset'],seed)
        h=hashlib.sha256(hashes(d['x']).tobytes()+d['y'].tobytes()+
            '\n'.join(map(str,d['uid'])).encode()).hexdigest()
        result[str(seed)]=dict(proxy_label_uid_sha256=h,rows=len(d['x']))
    return result


def capture_control():
    dpool=load_pool('cic_iomt_2024')
    rows=[]
    for seed in CONFIG['seeds']:
        d=prepare('cic_iomt_2024',dpool,CONFIG['sample_rows_per_dataset'],seed)
        captures=np.asarray(d['capture'])
        all_groups=sorted(set(captures))
        benign=[c for c in all_groups if c.startswith('Benign')]
        attack=[c for c in all_groups if not c.startswith('Benign')]
        rng=np.random.default_rng(seed)
        test_groups={benign[seed%len(benign)],*rng.choice(attack,max(1,round(.2*len(attack))),replace=False)}
        remain=np.flatnonzero(~np.isin(captures,list(test_groups)))
        test=np.flatnonzero(np.isin(captures,list(test_groups)))
        base=dict(dataset='cic_iomt_2024',seed=seed,split_mode='stratified_capture_holdout',
            calibration_scope='predictor groups within source-training captures; test captures fully disjoint')
        try:
            a,b=group_split(d['x'].iloc[remain],d['y'][remain],.15/.8,seed+1)
            train,cal=remain[a],remain[b]
            key=hashes(d['x'])
            train=train[~np.isin(key[train],key[np.r_[cal,test]])]
            cal=cal[~np.isin(key[cal],key[test])]
            if len(np.unique(d['y'][train]))<2 or len(np.unique(d['y'][cal]))<2 or len(np.unique(d['y'][test]))<2:
                raise ValueError('class coverage lost after capture holdout and predictor purging')
            assert not set(captures[train])&set(captures[test])
            transform=Transform('robust').fit(d['x'].iloc[train])
            x,y,z=[transform.apply(d['x'].iloc[idx],source=True) for idx in [train,cal,test]]
            for kind in ['lr','rf']:
                model=fit(kind,x,d['y'][train],y,d['y'][cal],seed)
                sc,_,_=get_scores(model,y);cutoff=source_threshold(d['y'][cal],sc)
                s,p,_=get_scores(model,z)
                rows.append(dict(**base,status='completed',model=kind,train_n=len(train),cal_n=len(cal),
                    test_n=len(test),train_benign=int((d['y'][train]==0).sum()),
                    cal_benign=int((d['y'][cal]==0).sum()),test_benign=int((d['y'][test]==0).sum()),
                    train_capture_n=len(set(captures[train])),test_capture_n=len(set(captures[test])),
                    predictor_overlap_test_rows=int(np.isin(key[test],key[train]).sum()),
                    **metrics(d['y'][test],(s>=cutoff).astype(int),s,p)))
        except ValueError as e:
            rows.append(dict(**base,status='not_estimable',reason=str(e)))
        pd.DataFrame(rows).to_csv(OUT/'capture_holdout_runs.csv',index=False)
    good=pd.DataFrame(rows).query("status=='completed'")
    if not good.empty:
        summaries(good,['dataset','split_mode','model'],VALUES).to_csv(OUT/'capture_holdout_summary.csv',index=False)


def baseline_costs():
    warnings.filterwarnings('ignore',message='X does not have valid feature names, but LGBMClassifier was fitted with feature names',category=UserWarning)
    pairs=[('cic_iot_2023','ton_iot'),('ton_iot','cic_iot_2023'),('cic_iomt_2024','cic_iot_2023')]
    pools={name:load_pool(name) for name in ['cic_iot_2023','ton_iot','cic_iomt_2024']}
    rows=[];seed=42
    for source_name,target_name in pairs:
        source=prepare(source_name,pools[source_name],CONFIG['sample_rows_per_dataset'],seed)
        target=prepare(target_name,pools[target_name],CONFIG['sample_rows_per_dataset'],seed)
        train,cal,_=partition(source,seed)
        adapt,evaluation=group_split(target['x'],target['y'],.5,seed+2,require_both=False)
        t=Transform('baseline').fit(source['x'].iloc[train],target['x'].iloc[adapt])
        x,y=[t.apply(source['x'].iloc[i],source=True) for i in [train,cal]]
        u,z=[t.apply(target['x'].iloc[i]) for i in [adapt,evaluation]]
        for kind in ['lr','rf','xgb','lgbm','mlp','tcn','dann']:
            model=fit(kind,x,source['y'][train],y,source['y'][cal],seed,target=u if kind=='dann' else None)
            rows.append(dict(source=source_name,target=target_name,seed=seed,model=kind,**benchmark(model,z)))
            pd.DataFrame(rows).to_csv(OUT/'baseline_cost_profile_seed42.csv',index=False)
        print('baseline cost profile',source_name,'->',target_name,flush=True)


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=['before-native','after-native','extras'],required=True)
    args=p.parse_args()
    if args.phase=='before-native':
        record=dict(data_sha256=hashlib.sha256((ROOT/'revision'/'code'/'data_revision.py').read_bytes()).hexdigest(),
            fingerprints=fingerprint())
        (OUT/'native_correction_before.json').write_text(json.dumps(record,indent=2),encoding='utf-8')
        directory=ROOT/'revision'/'inputs'/'computational_snapshot'
        directory.mkdir(exist_ok=True)
        shutil.copy2(ROOT/'revision'/'code'/'data_revision.py',directory/'data_revision_before_protocol_encoding.py')
    elif args.phase=='after-native':
        old=json.loads((OUT/'native_correction_before.json').read_text())
        current=fingerprint()
        if old['fingerprints']!=current:
            raise RuntimeError('Proxy records, labels, or IDs changed; do not reuse external results.')
        subprocess.run([sys.executable,'revision/code/run_revision.py','--stage','within'],cwd=ROOT,check=True)
        subprocess.run([sys.executable,'-m','unittest','discover','-s','revision/code','-p','test_*.py','-v'],cwd=ROOT,check=True)
        (OUT/'native_correction_verified.json').write_text(json.dumps(dict(proxy_labels_ids_unchanged=True,
            before=old['data_sha256'],after=hashlib.sha256((ROOT/'revision'/'code'/'data_revision.py').read_bytes()).hexdigest(),
            corrected_field='native TON representation now includes declared protocol encoding; all within controls rerun',
            within_returncode=0),indent=2),encoding='utf-8')
    else:
        capture_control()
        baseline_costs()
        (OUT/'extra_controls_verified.json').write_text(json.dumps(dict(
            source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            capture_holdout_completed=True,baseline_cost_profile_completed=True,
            profiling_seed=42,primary_detection_metrics_five_seeds=True),indent=2),encoding='utf-8')


if __name__=='__main__':
    main()
