"""Separate conditional raw-flow study; never changes the failed full-traffic gate."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import torch
from common import HERE,ROOT,digest,log,save
from features import FEATURES
from cost_models import SourceTransform,class_costs,teacher,student,probability,quantize
from operating_points import source_point,select_cost
from missing_label_bounds import metric_bounds
sys.path.insert(0,str(ROOT/'revision/code'))
from protocol import metrics,summaries

RUN=HERE/'labeled_raw_v3'
FAMILIES=['teacher','student_supervised','student_distilled','student_distilled_int8','lr','rf']
COMPONENTS=FAMILIES[:4]
SPEC=dict(version='labeled-raw-v3-conditional',
    estimand='Conditional performance among sampled flows with traceable binary label support; not full-traffic performance.',
    original_full_traffic_gate='Preserved separately, never overridden by this study.',
    selection='Source train/calibration only; source recall under empirical FPR<=0.01 or 0.05; source tie-breaks fixed.',
    preprocessing='Source-training median imputation, signed log1p, source-training standardization.',
    student='12-64-32-2 MLP, dropout 0.1; matched supervised/KD capacity; weighted CE/KL, T=3, hard fraction=0.25.',
    teacher='XGBoost 300 trees, depth6, lr0.05, subsample/colsample0.9; 4 training threads.',
    controls='LR C1 max_iter2000; RF200 trees; identical source cohorts and transform; all declared class costs.',
    quantization='Dynamic INT8; row-wise TorchScript scoring; source-only operating-point calibration.',
    ablation='At source FPR0.05, all four components share cost1 or the KD source-selected cost; each has its own original source threshold.',
    missingness='Unknown labels are never imputed. Sharp bounds permit every unknown class assignment, including nonrandom missingness.',
    uncertainty='Five seeds condition on fixed capture pools; not independent deployment-site confidence intervals.',
    chronology='Capture-disjoint roles are retained; capture files need not be independent acquisition sessions.',
    compression_scope='Desktop completed-flow inference only; not Raspberry Pi/Jetson evidence.')


def code_hashes():
    return {str(p.relative_to(ROOT)).replace('\\','/'):digest(p) for p in [Path(__file__),HERE/'cost_models.py',
        HERE/'features.py',HERE/'operating_points.py',HERE/'missing_label_bounds.py',
        ROOT/'revision/code/neural_revision.py',ROOT/'revision/code/protocol.py']}


def verify_freeze():
    path=RUN/'manifest.json';manifest=json.loads(path.read_text())
    if manifest['code_sha256']!=code_hashes():
        raise RuntimeError('Conditional-study code changed after registration; version a new study.')
    for relative,value in manifest['snapshot_sha256'].items():
        if digest(RUN/relative)!=value:
            raise AssertionError('Conditional-study data/config changed after its freeze.')
    return manifest,digest(path)


def register():
    if (RUN/'manifest.json').exists():
        verify_freeze();log('LABELED RAW V3 REGISTERED SNAPSHOT VERIFIED');return
    data=json.loads((HERE/'manifests/common_dataset.json').read_text())
    if data.get('label_schema_version')!='v3' or not data.get('packet_witness_manifest_sha256'):
        raise RuntimeError('Versioned label normalization and complete packet-witness processing must precede registration.')
    audit=json.loads((HERE/'results/prepared_audit.json').read_text())
    for part in audit['partitions']:
        path=HERE/'prepared'/part['dataset']/(part['role']+'.parquet')
        if digest(path)!=part['sha256']:
            raise AssertionError('Prepared physical audit does not describe the current data.')
    config=json.loads((HERE/'config.json').read_text());config['study']=SPEC
    for folder in ['data','models','results','figures']:
        (RUN/folder).mkdir(parents=True,exist_ok=True)
    snapshot={}
    for relative,key in [('manifests/label_schema_v3.json','label_schema_manifest_sha256'),
        ('results/all_packet_witnesses_v3.json','packet_witness_manifest_sha256'),
        ('manifests/official_event_sources.json','official_event_sources_sha256')]:
        if digest(HERE/relative)!=data[key]:
            raise AssertionError('An annotation provenance receipt differs from the prepared data.')
    provenance=['manifests/label_schema_v3.json','results/all_packet_witnesses_v3.json',
        'manifests/official_event_sources.json','manifests/capture_plan.json','results/prepared_audit.json',
        'results/source_integrity.json','results/tail_repair_v3.json']
    for relative in provenance:
        source=HERE/relative;destination=RUN/'provenance'/Path(relative).name
        destination.parent.mkdir(exist_ok=True);shutil.copy2(source,destination)
        snapshot[str(destination.relative_to(RUN)).replace('\\','/')]=digest(destination)
    for dataset in config['primary_pair']:
        for role in ['train','calibration','final']:
            source=HERE/'prepared'/dataset/(role+'.parquet')
            destination=RUN/'data'/dataset/(role+'.parquet');destination.parent.mkdir(exist_ok=True)
            shutil.copy2(source,destination);snapshot[str(destination.relative_to(RUN)).replace('\\','/')]=digest(destination)
    for source,name in [(HERE/'prepared/unresolved_labels.parquet','unknown.parquet'),
        (HERE/'manifests/common_dataset.json','full_traffic_manifest.json')]:
        destination=RUN/'data'/name;shutil.copy2(source,destination)
        snapshot[str(destination.relative_to(RUN)).replace('\\','/')]=digest(destination)
    save(RUN/'config.json',config);snapshot['config.json']=digest(RUN/'config.json')
    save(RUN/'manifest.json',dict(registered_utc=datetime.now(timezone.utc).isoformat(),specification=SPEC,
        code_sha256=code_hashes(),snapshot_sha256=snapshot,
        original_full_traffic_quality_gate_passed=data['quality_gate_passed'],
        original_full_traffic_label_coverage=data['ton_label_coverage'],
        original_requirement_unchanged=.95,conditional_study_only=True,
        common_model_performance_observed_before_registration=False,
        no_target_labels_in_training_or_policy_selection=True))
    log('LABELED RAW V3 REGISTERED: separate conditional estimand, full-traffic failure retained, no unknown labels imputed')


def partition(dataset,role,seed,cap=True):
    frame=pd.read_parquet(RUN/'data'/dataset/(role+'.parquet'))
    if cap and len(frame)>30000:
        frame=frame.sample(n=30000,random_state=seed)
    return frame.reset_index(drop=True)


def source_data(dataset,seed):
    train=partition(dataset,'train',seed);cal=partition(dataset,'calibration',seed)
    if set(train.capture_id)&set(cal.capture_id) or set(train.raw_identity)&set(cal.raw_identity):
        raise AssertionError('Source capture/observational-identity overlap.')
    if set(train.label.unique())!={0,1} or set(cal.label.unique())!={0,1}:
        raise ValueError('Both source classes are required.')
    ids=lambda frame:hashlib.sha256('|'.join(frame.flow_uid).encode()).hexdigest()
    return train,cal,dict(train_rows=len(train),calibration_rows=len(cal),train_uid_sha256=ids(train),
        calibration_uid_sha256=ids(cal),train_capture_ids=sorted(train.capture_id.unique()),
        calibration_capture_ids=sorted(cal.capture_id.unique()))


def train():
    manifest,freeze_sha=verify_freeze();config=json.loads((RUN/'config.json').read_text())
    if (RUN/'results/evaluation_start.json').exists():
        raise RuntimeError('No training changes after conditional final evaluation starts.')
    frozen=[]
    for seed in config['seeds']:
        for source in config['primary_pair']:
            directory=RUN/'models'/source/str(seed);directory.mkdir(parents=True,exist_ok=True)
            record=directory/'frozen.json'
            if record.exists():
                info=json.loads(record.read_text())
                if info['study_manifest_sha256']!=freeze_sha:
                    raise AssertionError('A source freeze belongs to a different study.')
                for name,value in info['artifact_sha256'].items():
                    if digest(directory/name)!=value:
                        raise AssertionError('A frozen source artifact changed.')
                frozen.append(record);log(f'LABELED RAW SOURCE CACHE VERIFIED {source}/seed{seed}');continue
            training,cal,audit=source_data(source,seed)
            transform=SourceTransform().fit(training[FEATURES]);x=transform.apply(training);v=transform.apply(cal)
            y=training.label.to_numpy();yv=cal.label.to_numpy()
            joblib.dump(transform,directory/'transform.joblib');artifacts={'transform.joblib':digest(directory/'transform.joblib')}
            candidates=[];history=[]
            for cost in config['benign_cost_grid']:
                start=time.monotonic();log(f'LABELED RAW TRAIN {source}/seed{seed}/cost{cost}: target labels unavailable to fitting')
                large=teacher(x,y,seed,cost)
                plain=student(x,y,v,yv,seed,cost,config['student'])
                distilled=student(x,y,v,yv,seed,cost,config['student'],large.predict_proba(x))
                weights=class_costs(y,cost)
                lr=LogisticRegression(C=1.,max_iter=2000,random_state=seed).fit(x,y,sample_weight=weights[y])
                rf=RandomForestClassifier(n_estimators=200,n_jobs=4,random_state=seed).fit(x,y,sample_weight=weights[y])
                rf.set_params(n_jobs=1)
                models=dict(teacher=large,student_supervised=plain,student_distilled=distilled,
                    student_distilled_int8=quantize(distilled),lr=lr,rf=rf)
                for family,model in models.items():
                    name=family+'_cost'+str(cost)+'.joblib';joblib.dump(model,directory/name);artifacts[name]=digest(directory/name)
                    scores=probability(model,v)
                    for alpha in config['source_fpr_targets']:
                        candidates.append(dict(family=family,candidate_id=name,benign_cost=cost,**source_point(yv,scores,alpha)))
                    history.append(dict(family=family,benign_cost=cost,parameters=getattr(model,'parameter_count',None),
                        training_history=getattr(model,'training_history',[])))
                log(f'LABELED RAW COST COMPLETE {source}/seed{seed}/cost{cost}: {time.monotonic()-start:.1f}s')
            selected=[]
            for family in FAMILIES:
                for alpha in config['source_fpr_targets']:
                    options=[{k:value for k,value in r.items() if k!='family'} for r in candidates if r['family']==family and r['fpr_limit']==alpha]
                    standard=next(r for r in options if r['benign_cost']==1)
                    for arm,point in [('standard',standard),('cost_sensitive',select_cost(options))]:
                        selected.append(dict(family=family,arm=arm,**point))
            kd=next(r for r in selected if r['family']=='student_distilled' and r['arm']=='cost_sensitive' and r['fpr_limit']==.05)
            matched=[]
            for arm,cost in [('standard_cost',1),('matched_distilled_cost',kd['benign_cost'])]:
                for family in COMPONENTS:
                    point=next(r for r in candidates if r['family']==family and r['benign_cost']==cost and r['fpr_limit']==.05)
                    matched.append(dict(arm=arm,**point))
            save(record,dict(source=source,seed=seed,study_manifest_sha256=freeze_sha,source_audit=audit,
                feature_columns=FEATURES,artifact_sha256=artifacts,candidates=candidates,selections=selected,
                matched_components=matched,training=history,target_metrics_used_for_selection=False,
                final_performance_observed_before_freeze=False))
            frozen.append(record);log(f'LABELED RAW SOURCE FROZEN {source}/seed{seed}: 30 artifacts, 24 source policies')
    save(RUN/'frozen_sources.json',dict(study_manifest_sha256=freeze_sha,
        files={str(p.relative_to(RUN)).replace('\\','/'):digest(p) for p in frozen},all_sources_frozen_before_final=True))
    log('LABELED RAW TRAINING COMPLETE: 10 source/seed bundles; final labels not used for fitting or selection')


def evaluate():
    manifest,freeze_sha=verify_freeze();config=json.loads((RUN/'config.json').read_text())
    source_manifest=json.loads((RUN/'frozen_sources.json').read_text())
    if len(source_manifest['files'])!=10 or source_manifest['study_manifest_sha256']!=freeze_sha:
        raise AssertionError('All ten matching source freezes are required before evaluation.')
    for name,value in source_manifest['files'].items():
        if digest(RUN/name)!=value:
            raise AssertionError('A source choice record changed before evaluation.')
    save(RUN/'results/evaluation_start.json',dict(frozen_sources_sha256=digest(RUN/'frozen_sources.json'),
        started_utc=datetime.now(timezone.utc).isoformat()))
    torch.set_num_threads(1)
    unknown_all=pd.read_parquet(RUN/'data/unknown.parquet')
    rows=[];bounds=[];components=[];groups=[]
    for name in source_manifest['files']:
        record=RUN/name;info=json.loads(record.read_text());directory=record.parent
        source=info['source'];seed=info['seed']
        for filename,value in info['artifact_sha256'].items():
            if digest(directory/filename)!=value:
                raise AssertionError('A source model changed before evaluation.')
        transform=joblib.load(directory/'transform.joblib');models={}
        for target in config['primary_pair']:
            frame=partition(target,'final',seed,cap=False);y=frame.label.to_numpy();x=transform.apply(frame)
            missing=unknown_all[unknown_all.dataset.eq(target)&unknown_all.role.eq('final')]
            xu=transform.apply(missing) if len(missing) else None
            score_cache={};unknown_cache={}
            policies=[dict(r,analysis='primary') for r in info['selections']]
            if source!=target:
                policies.extend(dict(r,analysis='matched_components') for r in info['matched_components'])
            for chosen in policies:
                filename=chosen['candidate_id']
                if filename not in models:
                    models[filename]=joblib.load(directory/filename)
                if filename not in score_cache:
                    score_cache[filename]=probability(models[filename],x)
                    unknown_cache[filename]=probability(models[filename],xu) if xu is not None else np.array([])
                scores=score_cache[filename];prediction=(scores>=chosen['threshold']).astype(int)
                values=metrics(y,prediction,scores,scores)
                base=dict(source=source,target=target,seed=seed,family=chosen['family'],arm=chosen['arm'],
                    benign_cost=chosen['benign_cost'],source_fpr_limit=chosen['fpr_limit'],threshold=chosen['threshold'],
                    calibration_recall=chosen['source_recall'],calibration_fpr=chosen['source_fpr'],
                    evaluation='within' if source==target else 'external',conditional_labeled_subcohort=True)
                if chosen['analysis']=='primary':
                    rows.append(dict(base,**values))
                else:
                    components.append(dict(base,**values))
                u1=int((unknown_cache[filename]>=chosen['threshold']).sum());u0=len(missing)-u1
                bounds.append(dict(base,analysis=chosen['analysis'],**metric_bounds(values['tn'],values['fp'],values['fn'],values['tp'],u1,u0)))
                if chosen['analysis']=='primary' and source!=target and chosen['arm']=='cost_sensitive' and chosen['fpr_limit']==.05 and chosen['family'] in ['teacher','student_distilled']:
                    errors=pd.DataFrame(dict(attack=frame.attack,label=y,prediction=prediction))
                    for (attack,label),group in errors.groupby(['attack','label']):
                        alarms=int(group.prediction.sum());n=len(group)
                        groups.append(dict(base,attack=attack,label=int(label),n=n,
                            tp=alarms if label==1 else 0,fn=n-alarms if label==1 else 0,
                            fp=alarms if label==0 else 0,tn=n-alarms if label==0 else 0))
            pd.DataFrame(rows).to_csv(RUN/'results/performance.csv',index=False)
            pd.DataFrame(bounds).to_csv(RUN/'results/missing_label_bounds.csv',index=False)
            log(f'LABELED RAW EVALUATED {source}->{target}/seed{seed}: {len(frame):,} labeled, {len(missing):,} unknown; no missing labels filled')
    data=pd.DataFrame(rows);component=pd.DataFrame(components)
    if len(data)!=480 or len(component)!=80:
        raise AssertionError('All 480 prespecified policies and 80 matched component results are required.')
    component.to_csv(RUN/'results/matched_components.csv',index=False)
    pd.DataFrame(groups).to_csv(RUN/'results/attack_errors.csv',index=False)
    summaries(data,['source','target','family','arm','source_fpr_limit','evaluation'],
        ['accuracy','macro_f1','balanced_accuracy','precision','recall','fpr','roc_auc','pr_auc']).to_csv(RUN/'results/summary.csv',index=False)
    save(RUN/'results/completed.json',dict(policies=len(data),matched_component_rows=len(component),seeds=config['seeds'],
        source_manifest_sha256=digest(RUN/'frozen_sources.json'),study_manifest_sha256=freeze_sha,
        conditional_labeled_subcohort_only=True,unknown_labels_imputed=False,
        original_full_traffic_quality_gate_passed=manifest['original_full_traffic_quality_gate_passed']))
    log('LABELED RAW EVALUATION COMPLETE: conditional results plus sharp full-sample bounds; not a full-traffic generalization claim')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['register','train','evaluate'],required=True)
    stage=parser.parse_args().stage
    register() if stage=='register' else train() if stage=='train' else evaluate()
