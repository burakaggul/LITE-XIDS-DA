"""Independent numeric, selection and missing-label-bound checks for raw v3."""
import json
import numpy as np
import pandas as pd
from common import digest,log,save
from labeled_raw_experiment import RUN,FAMILIES,verify_freeze,source_data
from operating_points import select_cost
from audit_results import rates
from missing_label_bounds import metric_bounds


def audit():
    manifest,study_sha=verify_freeze()
    frozen=json.loads((RUN/'frozen_sources.json').read_text())
    data=pd.read_csv(RUN/'results/performance.csv')
    keys=['source','target','seed','family','arm','source_fpr_limit']
    if len(data)!=480 or data[keys].duplicated().any() or set(data.family)!=set(FAMILIES):
        raise AssertionError('Missing or duplicate prespecified raw policies.')
    checked=rates(data)
    for actual,expected in [('fpr','fpr'),('recall','recall'),('precision','precision'),('macro_f1','macro_f1_binary_labels_0_1')]:
        np.testing.assert_allclose(data[actual],checked[expected],atol=1e-12,rtol=1e-12)
    if (data.calibration_fpr>data.source_fpr_limit+1e-12).any():
        raise AssertionError('Source empirical FPR constraint violated.')
    for name,value in frozen['files'].items():
        file=RUN/name
        if digest(file)!=value:
            raise AssertionError('A source choice record changed.')
        info=json.loads(file.read_text());directory=file.parent
        if info['study_manifest_sha256']!=study_sha or info['target_metrics_used_for_selection'] or info['final_performance_observed_before_freeze']:
            raise AssertionError('Invalid source selection provenance.')
        _,_,source_audit=source_data(info['source'],info['seed'])
        if source_audit!=info['source_audit']:
            raise AssertionError('Source cohort differs from its original source freeze.')
        for artifact,sha in info['artifact_sha256'].items():
            if digest(directory/artifact)!=sha:
                raise AssertionError('A frozen model/transform changed.')
        for selected in info['selections']:
            candidates=[{k:v for k,v in r.items() if k!='family'} for r in info['candidates'] if r['family']==selected['family'] and r['fpr_limit']==selected['fpr_limit']]
            expected=select_cost(candidates) if selected['arm']=='cost_sensitive' else next(c for c in candidates if c['benign_cost']==1)
            if any(selected[k]!=v for k,v in expected.items()):
                raise AssertionError('A policy was not reproduced from source-only candidates.')
            matched=data[(data.source==info['source'])&(data.seed==info['seed'])&data.family.eq(selected['family'])&data.arm.eq(selected['arm'])&data.source_fpr_limit.eq(selected['fpr_limit'])]
            if len(matched)!=2 or not matched.benign_cost.eq(selected['benign_cost']).all():
                raise AssertionError('Final policy differs from its source choice.')
            np.testing.assert_allclose(matched.threshold,selected['threshold'],rtol=1e-14,atol=0)
    components=pd.read_csv(RUN/'results/matched_components.csv')
    if len(components)!=80 or components.groupby(['source','seed','arm']).benign_cost.nunique().max()!=1:
        raise AssertionError('Matched component costs are not held fixed.')
    counted=rates(components)
    np.testing.assert_allclose(components.macro_f1,counted.macro_f1_binary_labels_0_1,rtol=1e-12,atol=1e-12)
    bounds=pd.read_csv(RUN/'results/missing_label_bounds.csv')
    if len(bounds)!=560 or bounds.missing_labels_imputed.any():
        raise AssertionError('Missing policy bounds or imputed unknown labels.')
    primary_bounds=bounds[bounds.analysis.eq('primary')]
    joined=data.merge(primary_bounds,on=keys,suffixes=('_observed','_bounds'),validate='one_to_one')
    for row in joined.itertuples():
        expected=metric_bounds(row.tn,row.fp,row.fn,row.tp,row.unknown_alarms,row.unknown_no_alarms)
        for metric in ['macro_f1','recall','fpr','balanced_accuracy','precision','accuracy']:
            for endpoint in ['lower','upper']:
                field=metric+'_'+endpoint
                np.testing.assert_allclose(getattr(row,field),expected[field],atol=1e-12,rtol=1e-12,equal_nan=True)
    errors=pd.read_csv(RUN/'results/attack_errors.csv')
    grouped=errors.groupby(['source','target','seed','family'])[['n','tn','fp','fn','tp']].sum().reset_index()
    selected=data[data.evaluation.eq('external')&data.arm.eq('cost_sensitive')&data.source_fpr_limit.eq(.05)&data.family.isin(['teacher','student_distilled'])]
    match=selected.merge(grouped,on=['source','target','seed','family'],suffixes=('_full','_groups'),validate='one_to_one')
    if len(match)!=20:
        raise AssertionError('Missing full known-label subtype totals.')
    for name in ['n','tn','fp','fn','tp']:
        np.testing.assert_array_equal(match[name+'_full'],match[name+'_groups'])
    save(RUN/'results/verified.json',dict(policies=480,matched_components=80,missing_label_bound_rows=560,
        all_confusion_metrics_independently_recomputed=True,all_source_choices_independently_reproduced=True,
        all_data_and_model_hashes_verified=True,all_source_cohorts_matched=True,all_20_subtype_totals_matched=True,
        source_bundles=len(frozen['files']),original_full_traffic_quality_gate_passed=manifest['original_full_traffic_quality_gate_passed'],
        conditional_scope=True,unknown_labels_imputed=False,study_manifest_sha256=study_sha))
    log('LABELED RAW AUDIT PASSED: 480 policies, 80 matched components, 560 sharp-bound rows, 10 source freezes; conditional scope retained')


if __name__=='__main__':
    audit()
