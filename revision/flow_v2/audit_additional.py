"""Independently verify diagnostic controls against their primary confusion counts."""
import json
import numpy as np
import pandas as pd
from common import HERE,digest,log,save
from audit_results import rates


def audit():
    ablation=pd.read_csv(HERE/'results/legacy_matched_component_ablation.csv')
    recomputed=rates(ablation)
    for observed,expected in [('fpr','fpr'),('recall','recall'),('precision','precision'),
        ('macro_f1','macro_f1_binary_labels_0_1')]:
        np.testing.assert_allclose(ablation[observed],recomputed[expected],atol=1e-12,rtol=1e-12)
    if len(ablation)!=80 or ablation.groupby(['source','seed','arm']).benign_cost.nunique().max()!=1:
        raise AssertionError('Matched-component controls are incomplete or unmatched.')
    for (source,seed),group in ablation.groupby(['source','seed']):
        path=HERE/'models/legacy'/source/str(seed)/'frozen.json';info=json.loads(path.read_text())
        kd=next(c for c in info['selections'] if c['family']=='student_distilled' and c['arm']=='cost_sensitive' and c['fpr_limit']==.05)
        if not group[group.arm=='matched_distilled_cost'].benign_cost.eq(kd['benign_cost']).all():
            raise AssertionError('Component control cost is not the frozen source choice.')
        for row in group.itertuples():
            point=next(c for c in info['candidates'] if c['family']==row.family and c['benign_cost']==row.benign_cost and c['fpr_limit']==.05)
            np.testing.assert_allclose(row.threshold,point['threshold'],rtol=1e-14,atol=0)
    primary=pd.read_csv(HERE/'results/performance_legacy.csv')
    selected=primary[(primary.evaluation=='external')&(primary.arm=='cost_sensitive')&
        (primary.source_fpr_limit==.05)&primary.family.isin(['teacher','student_distilled'])]
    subtypes=pd.read_csv(HERE/'results/legacy_external_attack_subtype_errors.csv')
    keys=['source','target','seed','family'];columns=['n','tn','fp','fn','tp']
    total=subtypes.groupby(keys)[columns].sum().reset_index()
    merged=selected.merge(total,on=keys,suffixes=('_primary','_subtypes'),validate='one_to_one')
    if len(merged)!=20:
        raise AssertionError('All external error-analysis cohorts must match their primary evaluations.')
    for name in columns:
        np.testing.assert_array_equal(merged[name+'_primary'],merged[name+'_subtypes'])
    shap=pd.read_csv(HERE/'results/legacy_external_shap.csv')
    if len(shap)!=240 or not shap.groupby(keys).size().eq(12).all():
        raise AssertionError('Missing feature explanations.')
    xai=json.loads((HERE/'results/legacy_xai_audit.json').read_text())
    if xai['local_additivity_max_residual']>1e-4:
        raise AssertionError('Explanation additivity tolerance exceeded.')
    cpu=pd.read_csv(HERE/'results/legacy_matched_backend_cpu.csv')
    if len(cpu)!=20 or cpu.groupby(['source','seed']).benign_cost.nunique().max()!=1 or cpu.backend.nunique()!=1:
        raise AssertionError('CPU backend/cost pairing is invalid.')
    if not cpu[['model_ms_median','model_ms_p95','state_dict_bytes','exported_script_bytes']].gt(0).all().all():
        raise AssertionError('Invalid runtime/size measurements.')
    save(HERE/'results/additional_diagnostic_audit.json',dict(
        all_80_component_confusion_metrics_recomputed=True,all_component_costs_and_thresholds_reproduced_from_source=True,
        all_20_subtype_confusion_totals_match_primary_results=True,external_shap_feature_rows=len(shap),
        all_20_cpu_profiles_same_backend_and_cost=True,
        interpretation='Historical diagnostic only. INT8 size reduction is not a runtime improvement; higher FPR or lost recall cannot be called successful transfer.',
        code_sha256=digest(__file__)))
    log('ADDITIONAL DIAGNOSTIC AUDIT PASS: 80 component results, 20 full subtype totals, 240 SHAP rows and 20 matched CPU profiles')


if __name__=='__main__':
    audit()
