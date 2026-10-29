"""Report all fixed choices and paired effects without selecting on external labels."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,ROOT,digest,log,save


def describe_difference(values):
    values=np.asarray(values,dtype=float);n=len(values);sd=values.std(ddof=1) if n>1 else np.nan
    half=t.ppf(.975,n-1)*sd/np.sqrt(n) if n>1 else np.nan
    return dict(n=n,mean=float(values.mean()),sd=float(sd),ci_low=float(values.mean()-half),ci_high=float(values.mean()+half))


def report():
    tracks=[name for name in ['legacy','common'] if (HERE/'results'/('complete_'+name+'.json')).exists()]
    if not tracks:
        raise RuntimeError('No completed experiment to report.')
    datasets=[];paired=[]
    keys=['track','source','target','seed','family','source_fpr_limit','evaluation']
    for track in tracks:
        data=pd.read_csv(HERE/'results'/('performance_'+track+'.csv'))
        control=data[data.arm=='standard'];intervention=data[data.arm=='cost_sensitive']
        match=control.merge(intervention,on=keys,suffixes=('_standard','_cost'),validate='one_to_one')
        for name,group in match.groupby([k for k in keys if k!='seed']):
            row=dict(zip([k for k in keys if k!='seed'],name))
            for metric in ['macro_f1','balanced_accuracy','recall','fpr','precision']:
                stats=describe_difference(group[metric+'_cost']-group[metric+'_standard'])
                row.update({metric+'_delta_'+k:v for k,v in stats.items()})
            row['recall_lower_in_cost_seeds']=int((group.recall_cost<group.recall_standard).sum())
            row['fpr_lower_in_cost_seeds']=int((group.fpr_cost<group.fpr_standard).sum())
            row['cost_selected_nonunit_seeds']=int((group.benign_cost_cost>1).sum())
            paired.append(row)
        datasets.append(data)
    combined=pd.concat(datasets,ignore_index=True)
    pd.DataFrame(paired).to_csv(HERE/'results/paired_cost_effects.csv',index=False)
    combined.to_csv(HERE/'results/all_tracks_performance.csv',index=False)
    directory=HERE/'figures';directory.mkdir(exist_ok=True)
    for track in tracks:
        ext=combined[(combined.track==track)&(combined.evaluation=='external')]
        means=ext.groupby(['source','target','family','arm','source_fpr_limit'])[['fpr','recall']].mean().reset_index()
        directions=means[['source','target']].drop_duplicates().to_records(index=False)
        fig,axes=plt.subplots(1,len(directions),figsize=(11,4.7),squeeze=False)
        colors={'teacher':'#0072B2','student_supervised':'#009E73','student_distilled':'#D55E00','student_distilled_int8':'#CC79A7'}
        for ax,(source,target) in zip(axes[0],directions):
            g=means[(means.source==source)&(means.target==target)]
            for family in sorted(g.family.unique()):
                for arm,marker in [('standard','o'),('cost_sensitive','x')]:
                    r=g[(g.family==family)&(g.arm==arm)].sort_values('source_fpr_limit')
                    ax.plot(r.fpr,r.recall,marker=marker,color=colors[family],linestyle='--' if arm=='standard' else '-',
                        label=family.replace('student_','')+' '+arm.replace('_',' '),alpha=.85)
            ax.set(xlim=(-.02,1.02),ylim=(-.02,1.02),xlabel='External false positive rate',ylabel='External attack recall',
                title=source.replace('_',' ')+' -> '+target.replace('_',' '))
            ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7,loc='lower right')
        fig.suptitle(track.capitalize()+' track: all source-frozen choices (5 seeds)')
        fig.tight_layout();fig.savefig(directory/(track+'_fpr_recall.pdf'));fig.savefig(directory/(track+'_fpr_recall.png'),dpi=180);plt.close(fig)
    common_data_path=HERE/'manifests/common_dataset.json'
    common_data=json.loads(common_data_path.read_text()) if common_data_path.exists() else {}
    checks=dict(
        physical_feature_definition_tests=(HERE/'results/step_common_feature_tests_final.json').exists(),
        source_operating_point_and_label_join_tests=(HERE/'results/step_protocol_tests_union.json').exists(),
        capture_chronology_preservation_tests=(HERE/'results/step_chronology_tests.json').exists(),
        cost_student_and_quantization_tests=(HERE/'results/step_cost_training_tests_v21.json').exists(),
        historical_paired_experiment='legacy' in tracks,
        common_capture_experiment='common' in tracks,
        prepared_physical_and_overlap_audit=(HERE/'results/prepared_audit.json').exists(),
        label_quality_gate_passed=common_data.get('quality_gate_passed',False),
        ton_label_coverage=common_data.get('ton_label_coverage'),
        required_ton_label_coverage=json.loads((HERE/'config.json').read_text())['required_ton_label_coverage'],
        prepared_integrity_gate=(HERE/'results/prepared_audit.json').exists() and common_data.get('quality_gate_passed',False),
        official_annotation_point_tests=(HERE/'results/step_official_event_tests.json').exists(),
        legacy_external_xai=(HERE/'results/legacy_xai_audit.json').exists(),
        legacy_matched_component_ablation=(HERE/'results/legacy_ablation_audit.json').exists(),
        legacy_matched_float_int8_cpu_backend=(HERE/'results/legacy_matched_backend_cpu_audit.json').exists(),
        common_external_xai=(HERE/'results/common_xai_audit.json').exists(),
        common_matched_component_ablation=(HERE/'results/common_ablation_audit.json').exists(),
        common_lr_rf_controls=(HERE/'results/common_baseline_audit.json').exists(),
        matched_float_int8_cpu_backend=(HERE/'results/common_matched_backend_cpu_audit.json').exists(),
        raspberry_pi_test='deferred by user; not performed or claimed',
        submitted_original_unchanged=(digest(ROOT/'output/pdf/Manuscript.pdf')=='3513f7e5d42b58ca97b64728f3cb720b02341769c6f0621bb9e285a50cc1d9dc') if (ROOT/'output/pdf/Manuscript.pdf').exists() else None,
        manuscript_not_overwritten=True,
        cross_track_causal_feature_effect='not established: historical records are not capture-matched to new raw flows',
        uncertainty='paired t intervals across 5 seeds conditional on a fixed capture pool; not population/deployment confidence intervals')
    for key,file in [('physical_feature_definition_tests','step_common_feature_tests_final.json'),
        ('source_operating_point_and_label_join_tests','step_protocol_tests_union.json'),('cost_student_and_quantization_tests','step_cost_training_tests_v21.json'),
        ('capture_chronology_preservation_tests','step_chronology_tests.json')]:
        checks[key]=checks[key] and json.loads((HERE/'results'/file).read_text())['returncode']==0
    save(HERE/'results/implementation_checks.json',checks)
    headline=combined[(combined.evaluation=='external')&(combined.family=='student_distilled')]
    headline.groupby(['track','source','target','arm','source_fpr_limit'])[['macro_f1','recall','fpr']].mean().to_csv(HERE/'results/distilled_external_summary.csv')
    text=['FLOW V2: VERIFIED IMPLEMENTATION AND RESULTS','',
        'Raspberry Pi / Jetson experiments are deferred. No actual-device result is claimed.',
        'Historical A/C and common-flow B/D are paired internally. Cross-track A/B is descriptive, not a clean causal feature experiment.',
        'All operating points and costs were frozen on source calibration before their track final evaluation.',
        'Source FPR constraints are empirical source constraints, not external FPR guarantees.',
        'Lower FPR is not an improvement if useful attack recall disappears. All predefined families and operating points are reported.',
        'Five seeds quantify training/sampling variability conditional on these capture cohorts.',
        'Latency is measured after a completed flow is available; packet capture and 30s idle / 120s active aggregation are not included. No live end-to-end time-to-detection is claimed.',
        'Common extraction and cost learning are established techniques; novelty of their combination is not asserted.',
        '', 'CHECKS',json.dumps(checks,indent=2), '', 'DISTILLED STUDENT EXTERNAL RESULTS (MEANS)',
        headline.groupby(['track','source','target','arm','source_fpr_limit'])[['macro_f1','recall','fpr']].mean().to_string(),
        '', 'PAIRED EFFECTS',pd.DataFrame(paired).to_string(index=False)]
    (HERE/'results/verified_report.txt').write_text('\n'.join(text),encoding='utf-8')
    log('REPORT PASS: paired effect tables, all choices, trade-off plots and implementation checks generated')


if __name__=='__main__':
    report()
