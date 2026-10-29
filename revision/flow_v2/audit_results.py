"""Recompute key metrics from confusion counts and verify source-only frozen choices."""
import argparse
import json
import numpy as np
import pandas as pd
from common import HERE,digest,log,save
from operating_points import select_cost


def rates(frame):
    d=frame.copy();tn=d.tn.to_numpy();fp=d.fp.to_numpy();fn=d.fn.to_numpy();tp=d.tp.to_numpy()
    safe=lambda a,b:np.divide(a,b,out=np.zeros(len(a),dtype=float),where=b>0)
    d['fpr']=safe(fp,tn+fp);d['recall']=safe(tp,fn+tp);d['precision']=safe(tp,tp+fp)
    d['macro_f1_binary_labels_0_1']=(safe(2*tp,2*tp+fp+fn)+safe(2*tn,2*tn+fp+fn))/2
    d['false_alarms_per_10000_benign']=10000*d.fpr
    return d


def audit(track):
    config=json.loads((HERE/'config.json').read_text())
    base=HERE/'results';data=pd.read_csv(base/('performance_'+track+'.csv'))
    keys=['source','target','seed','family','arm','source_fpr_limit']
    if len(data)!=320 or data[keys].duplicated().any():
        raise AssertionError('Expected all 320 unique prespecified results per track.')
    recomputed=rates(data)
    for name,calculated in [('fpr','fpr'),('recall','recall'),('precision','precision'),('macro_f1','macro_f1_binary_labels_0_1')]:
        np.testing.assert_allclose(data[name],recomputed[calculated],atol=1e-12,rtol=1e-12)
    if (data.calibration_fpr>data.source_fpr_limit+1e-12).any():
        raise AssertionError('Source FPR constraint was violated.')
    manifest=HERE/'manifests'/('frozen_'+track+'.json');frozen=json.loads(manifest.read_text())
    verified=0
    for file,sha in frozen['freeze_sha256'].items():
        if digest(HERE/file)!=sha:
            raise AssertionError('Source choice record modified after freeze.')
        d=json.loads((HERE/file).read_text());directory=(HERE/file).parent
        if d['external_metrics_used_for_selection'] or d['final_performance_observed_before_freeze']:
            raise AssertionError('Improper selection provenance.')
        for name,checksum in d['artifact_sha256'].items():
            if digest(directory/name)!=checksum:
                raise AssertionError('Frozen model modified.')
        for chosen in d['selections']:
            candidates=[{k:v for k,v in r.items() if k!='family'} for r in d['candidates']
                if r['family']==chosen['family'] and r['fpr_limit']==chosen['fpr_limit']]
            expected=select_cost(candidates) if chosen['arm']=='cost_sensitive' else next(r for r in candidates if r['benign_cost']==1)
            if any(chosen[k]!=v for k,v in expected.items()):
                raise AssertionError('Choice does not follow the source-only rule.')
            match=data[(data.source==d['source'])&(data.seed==d['seed'])&(data.family==chosen['family'])&
                (data.arm==chosen['arm'])&(data.source_fpr_limit==chosen['fpr_limit'])]
            np.testing.assert_allclose(match.threshold,chosen['threshold'],atol=0,rtol=1e-14)
            if not (match.benign_cost==chosen['benign_cost']).all():
                raise AssertionError('External result does not use its frozen cost.')
        verified+=1
    if verified!=10:
        raise AssertionError('Missing source/seed bundle.')
    group=rates(pd.read_csv(base/('per_group_'+track+'.csv')))
    group['metric_note']='Binary macro-F1 explicitly includes labels 0 and 1; class-specific rates and counts should be preferred for single-class capture groups.'
    group.to_csv(base/('per_group_'+track+'_binary_metrics.csv'),index=False)
    save(base/('result_audit_'+track+'.json'),dict(performance_rows=len(data),source_seed_bundles=verified,
        all_confusion_metrics_recomputed=True,all_choices_reproduced_from_source_only=True,
        all_frozen_model_hashes_verified=True,per_group_metric_convention_explicit=True,
        frozen_manifest_sha256=digest(manifest)))
    log('RESULT AUDIT PASS '+track+': 320 results, 10 source freezes, model hashes and source-only choices verified')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--track',choices=['common','legacy'],required=True);a=p.parse_args()
    audit(a.track)
