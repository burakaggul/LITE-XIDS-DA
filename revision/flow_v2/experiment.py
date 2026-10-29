"""Freeze source-only choices before final evaluation; historical and raw-flow tracks stay separate."""
import argparse
import io
import json
import platform
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import psutil
import torch
from common import HERE,ROOT,digest,log,save
from features import FEATURES
from cost_models import SourceTransform,teacher,student,probability,quantize,neural_bytes
from operating_points import source_point,select_cost

sys.path.insert(0,str(ROOT/'revision/code'))
from protocol import group_split,metrics,summaries
from data_revision import load_pool,prepare

FAMILIES=['teacher','student_supervised','student_distilled','student_distilled_int8']
METRICS=['macro_f1','balanced_accuracy','precision','recall','fpr','roc_auc','pr_auc',
    'false_alarms_per_10000','false_alarms_per_10000_benign','alarm_rate']
PROTOCOL=dict(preprocessing='source-train-only median imputation, signed log1p, source-train StandardScaler',
    amendment='v2.1 before any final scores: batch-one invariant INT8 inference and source-only recalibration of previously trained legacy weights',
    teacher=dict(n_estimators=300,max_depth=6,learning_rate=.05,subsample=.9,colsample_bytree=.9,threads=4),
    class_weight='inverse source training frequency times benign cost; mean training weight normalized to one',
    student='existing MLP: 12 -> 64 ReLU Dropout(0.1) -> 32 ReLU -> 2; identical supervised and KD capacity',
    loss='training-class-weighted hard CE and per-row soft KL; T=3, hard fraction=0.25',
    early_stopping='source calibration weighted hard CE, max20, patience4; same calibration also selects operating points',
    quantization='PyTorch dynamic qint8 Linear; scripted row-wise execution fixes activation calibration to one record regardless of outer batch; operating point calibrated on source only',
    historical='same legacy pool sample size and grouped source/final split as previous revision; target grouped half previously examined, diagnostic only',
    common='fixed capture-disjoint train/calibration/final roles; per-source train/cal cap30000; final never fitted',
    inference='single CPU thread, 30 warmups, 200 timed model-only batch1 calls and 50 preprocessing+model calls',
    comparison='A/C legacy paired; B/D common paired; cross-track contrasts are descriptive, not capture-matched causal effects',
    success='report both external recall and FPR; reducing alarms by predicting everything benign is not a successful IDS')


def code_hashes():
    return {p.name:digest(p) for p in [HERE/'cost_models.py',HERE/'experiment.py',HERE/'operating_points.py',
        HERE/'features.py',ROOT/'revision/code/neural_revision.py',ROOT/'revision/code/protocol.py',ROOT/'revision/code/data_revision.py']}


def lock_protocol():
    path=HERE/'manifests/training_protocol_v21.json'
    payload=dict(protocol=PROTOCOL,config_sha256=digest(HERE/'config.json'),code_sha256=code_hashes())
    if path.exists() and json.loads(path.read_text())!=payload:
        raise RuntimeError('Training code/protocol changed; preserve old runs and explicitly version a new experiment.')
    if not path.exists():
        save(path,payload);log('FROZEN training protocol/code before model fitting')
    return digest(path)


def legacy_pool(name,seed):
    return prepare(name,load_pool(name),30000,seed)


def legacy_indices(data,seed):
    remaining,final=group_split(data['x'],data['y'],.2,seed)
    a,b=group_split(data['x'].iloc[remaining],data['y'][remaining],.15/.8,seed+1)
    return remaining[a],remaining[b],final


def common_partition(name,role,seed,cap=True):
    frame=pd.read_parquet(HERE/'prepared'/name/(role+'.parquet'))
    if cap and len(frame)>30000:
        frame=frame.sample(n=30000,random_state=seed)
    return frame.reset_index(drop=True)


def source_data(track,name,seed):
    if track=='common':
        audit=json.loads((HERE/'manifests/common_dataset.json').read_text())
        if not audit['quality_gate_passed']:
            raise RuntimeError('Common-flow labels failed the declared data quality gate.')
        train=common_partition(name,'train',seed);cal=common_partition(name,'calibration',seed)
        if set(train.capture_id)&set(cal.capture_id) or set(train.raw_identity)&set(cal.raw_identity):
            raise AssertionError('Source capture/raw-identity overlap.')
        return train[FEATURES],train.label.to_numpy(),cal[FEATURES],cal.label.to_numpy(),dict(
            train_ids_hash=__import__('hashlib').sha256('|'.join(train.flow_uid).encode()).hexdigest(),
            calibration_ids_hash=__import__('hashlib').sha256('|'.join(cal.flow_uid).encode()).hexdigest(),
            train_captures=sorted(train.capture_id.unique()),calibration_captures=sorted(cal.capture_id.unique()))
    data=legacy_pool(name,seed);a,b,_=legacy_indices(data,seed)
    return data['x'].iloc[a],data['y'][a],data['x'].iloc[b],data['y'][b],dict(
        legacy_audit=data['audit'],train_ids_hash=__import__('hashlib').sha256('|'.join(data['uid'][a]).encode()).hexdigest(),
        calibration_ids_hash=__import__('hashlib').sha256('|'.join(data['uid'][b]).encode()).hexdigest())


def train_source(track,name,seed,config,protocol_hash):
    directory=HERE/'models'/track/name/str(seed);directory.mkdir(parents=True,exist_ok=True)
    frozen=directory/'frozen.json'
    if frozen.exists():
        old=json.loads(frozen.read_text())
        if old['protocol_sha256']!=protocol_hash:
            raise RuntimeError('Stored model protocol mismatch.')
        for filename,value in old['artifact_sha256'].items():
            if digest(directory/filename)!=value:
                raise RuntimeError('Stored model artifact checksum mismatch.')
        log(f'CACHE VERIFIED frozen {track}/{name}/seed{seed}');return
    x,y,xv,yv,audit=source_data(track,name,seed)
    if set(np.unique(y))!={0,1} or set(np.unique(yv))!={0,1}:
        raise ValueError('Source training/calibration lacks a class.')
    transform=SourceTransform().fit(x);xt=transform.apply(x);xc=transform.apply(xv)
    joblib.dump(transform,directory/'transform.joblib')
    candidate_rows=[];training_rows=[];artifacts={'transform.joblib':digest(directory/'transform.joblib')}
    for cost in config['benign_cost_grid']:
        start=time.perf_counter();log(f'TRAIN {track}/{name}/seed{seed}: benign cost={cost}')
        large=teacher(xt,y,seed,cost)
        soft=large.predict_proba(xt)
        plain=student(xt,y,xc,yv,seed,cost,config['student'])
        distilled=student(xt,y,xc,yv,seed,cost,config['student'],soft)
        models=dict(teacher=large,student_supervised=plain,student_distilled=distilled,
            student_distilled_int8=quantize(distilled))
        for family,model in models.items():
            filename=f'{family}_cost{cost}.joblib';joblib.dump(model,directory/filename)
            artifacts[filename]=digest(directory/filename)
            scores=probability(model,xc)
            for alpha in config['source_fpr_targets']:
                candidate_rows.append(dict(family=family,**source_point(yv,scores,alpha),
                    candidate_id=filename,benign_cost=cost))
            training_rows.append(dict(family=family,benign_cost=cost,
                parameter_count=getattr(model,'parameter_count',None),
                epochs=len(getattr(model,'training_history',[])),
                history=getattr(model,'training_history',[])))
        log(f'TRAIN PASS {track}/{name}/seed{seed}/cost{cost}: {time.perf_counter()-start:.1f}s; source-only calibration')
    selected=[]
    for family in FAMILIES:
        for alpha in config['source_fpr_targets']:
            choices=[{k:v for k,v in r.items() if k!='family'} for r in candidate_rows if r['family']==family and r['fpr_limit']==alpha]
            standard=next(c for c in choices if c['benign_cost']==1)
            for arm,choice in [('standard',standard),('cost_sensitive',select_cost(choices))]:
                selected.append(dict(family=family,arm=arm,**choice))
    save(frozen,dict(track=track,source=name,seed=seed,protocol_sha256=protocol_hash,
        feature_columns=transform.columns,source_train_rows=len(y),source_calibration_rows=len(yv),
        source_audit=audit,artifact_sha256=artifacts,candidates=candidate_rows,selections=selected,
        training=training_rows,external_metrics_used_for_selection=False,
        final_performance_observed_before_freeze=False))
    log(f'FREEZE PASS {track}/{name}/seed{seed}: all source operating points and artifact hashes locked')


def train(track):
    config=json.loads((HERE/'config.json').read_text());protocol_hash=lock_protocol()
    for seed in config['seeds']:
        for name in config['primary_pair']:
            train_source(track,name,seed,config,protocol_hash)
    freezes=[HERE/'models'/track/name/str(seed)/'frozen.json' for seed in config['seeds'] for name in config['primary_pair']]
    save(HERE/'manifests'/('frozen_'+track+'.json'),dict(protocol_sha256=protocol_hash,
        final_evaluation_started=False,freeze_sha256={str(p.relative_to(HERE)):digest(p) for p in freezes}))
    log('TRAIN COMPLETE '+track+': all 10 source/seed freezes available; final evaluation not run by training stage')


def recalibrate_legacy():
    torch.set_num_threads(1)
    config=json.loads((HERE/'config.json').read_text());protocol_hash=lock_protocol()
    if (HERE/'results/evaluation_start_legacy.json').exists():
        raise RuntimeError('Recalibration is forbidden after final evaluation has started.')
    archive=HERE/'archive/batch_calibration_v20'
    frozen_paths=[]
    for seed in config['seeds']:
        for source in config['primary_pair']:
            directory=HERE/'models/legacy'/source/str(seed)
            old_file=archive/'models/legacy'/source/str(seed)/'frozen.json'
            info=json.loads(old_file.read_text())
            for filename,sha in info['artifact_sha256'].items():
                if digest(directory/filename)!=sha:
                    raise RuntimeError('Previously source-trained weights were modified.')
            _,_,xv,yv,audit=source_data('legacy',source,seed)
            if audit['calibration_ids_hash']!=info['source_audit']['calibration_ids_hash']:
                raise RuntimeError('Source calibration records changed.')
            transform=joblib.load(directory/'transform.joblib');x=transform.apply(xv)
            candidates=[]
            for cost in config['benign_cost_grid']:
                for family in FAMILIES:
                    filename=f'{family}_cost{cost}.joblib';model=joblib.load(directory/filename)
                    scores=probability(model,x)
                    for alpha in config['source_fpr_targets']:
                        candidates.append(dict(family=family,candidate_id=filename,benign_cost=cost,
                            **source_point(yv,scores,alpha)))
            chosen=[]
            for family in FAMILIES:
                for alpha in config['source_fpr_targets']:
                    choices=[{k:v for k,v in row.items() if k!='family'} for row in candidates
                        if row['family']==family and row['fpr_limit']==alpha]
                    standard=next(c for c in choices if c['benign_cost']==1)
                    for arm,choice in [('standard',standard),('cost_sensitive',select_cost(choices))]:
                        chosen.append(dict(family=family,arm=arm,**choice))
            info['weights_training_protocol_sha256']=info['protocol_sha256']
            info['protocol_sha256']=protocol_hash;info['candidates']=candidates;info['selections']=chosen
            info['source_recalibration']='v2.1 row-wise INT8; no weight refit, no final/target scores examined'
            save(directory/'frozen.json',info);frozen_paths.append(directory/'frozen.json')
            log(f'RECALIBRATION FREEZE PASS legacy/{source}/seed{seed}: source-only, same trained weights')
    save(HERE/'manifests/frozen_legacy.json',dict(protocol_sha256=protocol_hash,final_evaluation_started=False,
        freeze_sha256={str(p.relative_to(HERE)):digest(p) for p in frozen_paths},
        pre_final_amendment='v2.1 batch-one invariant INT8 inference; v2.0 records archived'))


def evaluation_data(track,name,seed,within):
    if track=='common':
        frame=common_partition(name,'final',seed,cap=False)
        return frame[FEATURES],frame.label.to_numpy(),frame.capture_id.to_numpy()
    data=legacy_pool(name,seed)
    ids=legacy_indices(data,seed)[2] if within else group_split(data['x'],data['y'],.5,seed+2,require_both=False)[1]
    return data['x'].iloc[ids],data['y'][ids],data['attack'][ids]


def benchmark(model,transform,frame):
    torch.set_num_threads(1)
    if hasattr(model,'set_params'):
        model.set_params(n_jobs=1)
    row=frame.iloc[:1];x=transform.apply(row)
    for _ in range(30):
        probability(model,x)
    model_ms=[];pipeline_ms=[]
    for _ in range(200):
        start=time.perf_counter_ns();probability(model,x);model_ms.append((time.perf_counter_ns()-start)/1e6)
    for _ in range(50):
        start=time.perf_counter_ns();probability(model,transform.apply(row));pipeline_ms.append((time.perf_counter_ns()-start)/1e6)
    process=psutil.Process()
    stream=io.BytesIO();joblib.dump(model,stream)
    return dict(model_ms_median=float(np.median(model_ms)),model_ms_p95=float(np.quantile(model_ms,.95)),
        pipeline_ms_median=float(np.median(pipeline_ms)),pipeline_ms_p95=float(np.quantile(pipeline_ms,.95)),
        serialized_bytes=len(stream.getvalue()),state_dict_bytes=neural_bytes(model) if isinstance(model,torch.nn.Module) else None,
        process_rss_bytes=process.memory_info().rss,parameter_count=getattr(model,'parameter_count',None),
        actual_device='desktop CPU only; concurrent data processing may affect timing; RSS is process total, not isolated model memory')


def evaluate(track):
    config=json.loads((HERE/'config.json').read_text());lock_protocol()
    frozen_path=HERE/'manifests'/('frozen_'+track+'.json')
    frozen=json.loads(frozen_path.read_text())
    for path,value in frozen['freeze_sha256'].items():
        if digest(HERE/path)!=value:
            raise RuntimeError('Source freeze was modified before final evaluation.')
    outputs=[];groups=[];cpu=[]
    save(HERE/'results'/('evaluation_start_'+track+'.json'),dict(frozen_sha256=digest(frozen_path),
        started_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat()))
    for seed in config['seeds']:
        for source in config['primary_pair']:
            directory=HERE/'models'/track/source/str(seed)
            info=json.loads((directory/'frozen.json').read_text())
            for filename,value in info['artifact_sha256'].items():
                if digest(directory/filename)!=value:
                    raise RuntimeError('Frozen artifact changed before final evaluation.')
            transform=joblib.load(directory/'transform.joblib')
            cache={};benchmarked=set()
            for target in config['primary_pair']:
                frame,y,capture=evaluation_data(track,target,seed,source==target)
                x=transform.apply(frame)
                score_cache={}
                for selection in info['selections']:
                    filename=selection['candidate_id']
                    if filename not in cache:
                        cache[filename]=joblib.load(directory/filename)
                    model=cache[filename]
                    if filename not in score_cache:
                        score_cache[filename]=probability(model,x)
                    scores=score_cache[filename]
                    prediction=(scores>=selection['threshold']).astype(int)
                    base=dict(track=track,source=source,target=target,seed=seed,
                        family=selection['family'],arm=selection['arm'],source_fpr_limit=selection['fpr_limit'],
                        benign_cost=selection['benign_cost'],threshold=selection['threshold'],
                        calibration_recall=selection['source_recall'],calibration_fpr=selection['source_fpr'],
                        evaluation='within' if source==target else 'external')
                    outputs.append(dict(base,**metrics(y,prediction,scores,scores)))
                    for group in sorted(set(capture)):
                        mask=capture==group
                        groups.append(dict(base,group=group,**metrics(y[mask],prediction[mask],scores[mask],scores[mask])))
                    if filename not in benchmarked:
                        cpu.append(dict(track=track,source=source,seed=seed,family=selection['family'],
                            benign_cost=selection['benign_cost'],**benchmark(model,transform,frame)))
                        benchmarked.add(filename)
                log(f'EVALUATE {track}/{source}->{target}/seed{seed}: frozen selections only, {len(y):,} final rows')
                pd.DataFrame(outputs).to_csv(HERE/'results'/('performance_'+track+'.csv'),index=False)
    data=pd.DataFrame(outputs)
    keys=['track','source','target','family','arm','source_fpr_limit','evaluation']
    summaries(data,keys,METRICS).to_csv(HERE/'results'/('summary_'+track+'.csv'),index=False)
    pd.DataFrame(groups).to_csv(HERE/'results'/('per_group_'+track+'.csv'),index=False)
    pd.DataFrame(cpu).to_csv(HERE/'results'/('cpu_'+track+'.csv'),index=False)
    save(HERE/'results'/('complete_'+track+'.json'),dict(rows=len(data),seeds=config['seeds'],
        frozen_sha256=digest(frozen_path),source_fpr_is_not_target_fpr_guarantee=True,
        uncertainty='5 training/sampling seeds conditional on fixed capture cohort, not independent deployments',
        versions=dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,
            sklearn=__import__('sklearn').__version__,xgboost=__import__('xgboost').__version__),
        historical_diagnostic_only=track=='legacy'))
    log('EVALUATION COMPLETE '+track+': '+str(len(data))+' paired operating-point results')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['legacy','common'],required=True)
    p.add_argument('--stage',choices=['train','evaluate','recalibrate'],required=True);a=p.parse_args()
    if a.stage=='recalibrate':
        if a.track!='legacy':
            raise ValueError('Only the archived legacy pre-final calibration needs migration.')
        recalibrate_legacy()
    elif a.stage=='train':
        train(a.track)
    else:
        evaluate(a.track)
