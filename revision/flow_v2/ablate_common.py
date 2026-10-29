"""Matched-cost component controls, derived exclusively from pre-final source choices."""
import argparse
import json
import joblib
import numpy as np
import pandas as pd
from common import HERE,digest,log,save
from cost_models import probability
from experiment import FAMILIES,evaluation_data
from protocol import metrics,summaries


def run(track='common'):
    config=json.loads((HERE/'config.json').read_text())
    if not (HERE/'results'/('result_audit_'+track+'.json')).exists():
        raise RuntimeError('Independent primary result audit is required before component analysis.')
    manifest=HERE/'manifests'/('frozen_'+track+'.json');frozen=json.loads(manifest.read_text())
    choices=[]
    for path,value in frozen['freeze_sha256'].items():
        if digest(HERE/path)!=value:
            raise AssertionError('Source choice record was modified.')
        info=json.loads((HERE/path).read_text())
        kd=next(r for r in info['selections'] if r['family']=='student_distilled' and r['arm']=='cost_sensitive' and r['fpr_limit']==.05)
        for arm,cost in [('standard_cost',1),('matched_distilled_cost',kd['benign_cost'])]:
            for family in FAMILIES:
                point=next(r for r in info['candidates'] if r['family']==family and r['benign_cost']==cost and r['fpr_limit']==.05)
                choices.append(dict(source=info['source'],seed=info['seed'],arm=arm,**point))
    record=HERE/'manifests'/('fixed_cost_ablation_'+track+'_frozen.json')
    save(record,dict(choices=choices,source_manifest_sha256=digest(manifest),
        rule='All four model families share the distilled student source-selected cost at source FPR<=0.05; each operating point comes from the original frozen source calibration.',
        track=track,analysis_only=True,posthoc_component_analysis=True,
        not_an_independent_new_final_test=True,target_metrics_used_for_selection=False))
    primary=pd.read_csv(HERE/'results'/('performance_'+track+'.csv'));results=[]
    for source in config['primary_pair']:
        target=next(n for n in config['primary_pair'] if n!=source)
        for seed in config['seeds']:
            directory=HERE/'models'/track/source/str(seed)
            info=json.loads((directory/'frozen.json').read_text());transform=joblib.load(directory/'transform.joblib')
            frame,y,_=evaluation_data(track,target,seed,False);x=transform.apply(frame);cache={}
            for choice in [r for r in choices if r['source']==source and r['seed']==seed]:
                filename=choice['candidate_id']
                if digest(directory/filename)!=info['artifact_sha256'][filename]:
                    raise AssertionError('Matched control model changed.')
                reuse=primary[(primary.source==source)&(primary.target==target)&(primary.seed==seed)&
                    (primary.family==choice['family'])&(primary.source_fpr_limit==.05)&
                    (primary.benign_cost==choice['benign_cost'])&np.isclose(primary.threshold,choice['threshold'],atol=1e-12,rtol=1e-12)]
                base=dict(track=track,source=source,target=target,seed=seed,arm=choice['arm'],family=choice['family'],
                    benign_cost=choice['benign_cost'],source_fpr_limit=.05,threshold=choice['threshold'])
                if len(reuse):
                    r=reuse.iloc[0]
                    values={k:r[k] for k in ['macro_f1','balanced_accuracy','precision','recall','fpr','roc_auc','pr_auc','tn','fp','fn','tp','n']}
                    values['reused_identical_primary_prediction']=True
                else:
                    if filename not in cache:
                        model=joblib.load(directory/filename);cache[filename]=probability(model,x)
                    scores=cache[filename];values=metrics(y,(scores>=choice['threshold']).astype(int),scores,scores)
                    values['reused_identical_primary_prediction']=False
                results.append(dict(base,**values))
            log(f'MATCHED ABLATION PASS {track}/{source}->{target}/seed{seed}: fixed-cost teacher, supervised, KD and INT8 controls; diagnostic analysis, not a fresh final test')
    data=pd.DataFrame(results)
    if len(data)!=80:
        raise AssertionError('All 80 matched controls are required.')
    data.to_csv(HERE/'results'/(track+'_matched_component_ablation.csv'),index=False)
    summaries(data,['source','target','arm','family'],['macro_f1','recall','fpr','balanced_accuracy']).to_csv(
        HERE/'results'/(track+'_matched_component_ablation_summary.csv'),index=False)
    save(HERE/'results'/(track+'_ablation_audit.json'),dict(track=track,rows=len(data),all_costs_matched_within_seed_arm=True,
        source_only_policy_record_sha256=digest(record),not_independent_test=True,
        posthoc_component_analysis=True,code_sha256=digest(__file__),
        caveat='Component effects are paired within their own cohort and fixed source-selected cost. Legacy features are historical proxies, not physically matched raw flows. Five seeds are conditional resampling/training variability.'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--track',choices=['common','legacy'],default='common')
    run(parser.parse_args().track)
