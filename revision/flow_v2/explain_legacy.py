"""External explanations of existing diagnostic models, not a new validation test."""
import json
import joblib
import numpy as np
import pandas as pd
import shap
import torch
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,digest,log,save
from experiment import source_data,evaluation_data
from explain_common import FrozenPredictor


def explain():
    if not (HERE/'results/result_audit_legacy.json').exists():
        raise RuntimeError('Verified legacy final results are required.')
    config=json.loads((HERE/'config.json').read_text())
    frozen=HERE/'manifests/frozen_legacy.json';manifest=json.loads(frozen.read_text())
    for file,value in manifest['freeze_sha256'].items():
        if digest(HERE/file)!=value:
            raise AssertionError('Frozen choice record changed.')
    torch.set_num_threads(1)
    importance=[];examples=[];subtypes=[];pdp=[];residuals=[]
    for source in config['primary_pair']:
        target=next(name for name in config['primary_pair'] if name!=source)
        queries,query_y,_=evaluation_data('legacy',target,42,False)
        query_ids=np.random.default_rng(42).choice(len(queries),min(64,len(queries)),replace=False)
        fixed_queries=queries.iloc[query_ids]
        for seed in config['seeds']:
            directory=HERE/'models/legacy'/source/str(seed)
            info=json.loads((directory/'frozen.json').read_text())
            if digest(directory/'transform.joblib')!=info['artifact_sha256']['transform.joblib']:
                raise AssertionError('Frozen source transform changed.')
            transform=joblib.load(directory/'transform.joblib');columns=transform.columns
            training,_,_,_,audit=source_data('legacy',source,seed)
            if audit!=info['source_audit']:
                raise AssertionError('Source explanation background cohort changed.')
            raw_background=training.sample(n=min(64,len(training)),random_state=42)
            background=pd.DataFrame(transform.imputer.transform(raw_background),columns=columns)
            final,labels,groups=evaluation_data('legacy',target,seed,False)
            for family in ['teacher','student_distilled']:
                choice=next(c for c in info['selections'] if c['family']==family and c['arm']=='cost_sensitive' and c['fpr_limit']==.05)
                file=directory/choice['candidate_id']
                if digest(file)!=info['artifact_sha256'][file.name]:
                    raise AssertionError('Frozen explanation model changed.')
                predictor=FrozenPredictor(joblib.load(file),transform)
                predict=lambda a:predictor.predict_proba(a)[:,1]
                query=pd.DataFrame(transform.imputer.transform(fixed_queries),columns=columns)
                explainer=shap.PermutationExplainer(predict,background.to_numpy(),feature_names=columns,seed=seed)
                values=explainer(query.to_numpy(),max_evals=2*(2*len(columns)+1),silent=True)
                scores=predict(query.to_numpy())
                residual=float(np.max(np.abs(values.base_values+values.values.sum(1)-scores)))
                if residual>1e-4:
                    raise AssertionError('Probability SHAP local additivity failed.')
                residuals.append(residual)
                for feature,value in zip(columns,np.abs(values.values).mean(0)):
                    importance.append(dict(source=source,target=target,seed=seed,family=family,feature=feature,
                        mean_abs_shap_probability=float(value),benign_cost=choice['benign_cost'],source_fpr_limit=.05))
                for i in range(len(query)):
                    ranking=np.argsort(-np.abs(values.values[i]))[:3]
                    examples.append(dict(source=source,target=target,seed=seed,family=family,query_position=int(query_ids[i]),
                        label=int(query_y[query_ids[i]]),attack_probability=float(scores[i]),threshold=choice['threshold'],
                        alarm=int(scores[i]>=choice['threshold']),top_features=';'.join(columns[j] for j in ranking),
                        signed_contributions=';'.join(str(float(values.values[i,j])) for j in ranking)))
                full=predict(final)>=choice['threshold']
                errors=pd.DataFrame(dict(group=groups,label=labels,alarm=full.astype(int)))
                for (group,label),g in errors.groupby(['group','label']):
                    alarms=int(g.alarm.sum());n=len(g)
                    subtypes.append(dict(source=source,target=target,seed=seed,family=family,group=group,label=int(label),n=n,
                        tp=alarms if label==1 else 0,fn=n-alarms if label==1 else 0,
                        fp=alarms if label==0 else 0,tn=n-alarms if label==0 else 0))
                if seed==42:
                    for feature in ['duration','packet_total','bytes_mean']:
                        result=partial_dependence(predictor,query,features=[columns.index(feature)],
                            method='brute',kind='average',grid_resolution=12)
                        for grid,value in zip(result['grid_values'][0],result['average'][0]):
                            pdp.append(dict(source=source,target=target,family=family,feature=feature,
                                feature_value=float(grid),mean_attack_probability=float(value)))
                log(f'LEGACY EXTERNAL XAI PASS {source}->{target}/seed{seed}/{family}: 64 fixed external queries; additivity residual={residual:.3g}')
    data=pd.DataFrame(importance);between=[]
    for keys,g in data.groupby(['source','target','family']):
        for a in config['seeds']:
            for b in config['seeds']:
                if b<=a:
                    continue
                x=g[g.seed==a].set_index('feature').mean_abs_shap_probability
                y=g[g.seed==b].set_index('feature').mean_abs_shap_probability.reindex(x.index)
                sx=set(x.nlargest(5).index);sy=set(y.nlargest(5).index)
                between.append(dict(zip(['source','target','family'],keys),seed_a=a,seed_b=b,
                    top5_jaccard=len(sx&sy)/len(sx|sy),rank_spearman=float(spearmanr(x,y).statistic)))
    for name,table in [('shap',data),('shap_seed_stability',pd.DataFrame(between)),
        ('error_explanations',pd.DataFrame(examples)),('attack_subtype_errors',pd.DataFrame(subtypes)),('pdp',pd.DataFrame(pdp))]:
        table.to_csv(HERE/'results'/('legacy_external_'+name+'.csv'),index=False)
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for (source,target,family),group in data.groupby(['source','target','family']):
        stats=group.groupby('feature').mean_abs_shap_probability.agg(['mean','std']).sort_values('mean')
        fig,ax=plt.subplots(figsize=(7.2,4.8))
        ax.barh(stats.index,stats['mean'],xerr=stats['std'],color='#0072B2')
        ax.set(xlabel='Mean absolute SHAP contribution (attack probability)',
            title='Legacy diagnostic: '+source+' -> '+target+' / '+family)
        fig.tight_layout()
        stem='legacy_shap_'+source+'_'+family
        fig.savefig(out/(stem+'.pdf'));fig.savefig(out/(stem+'.png'),dpi=170);plt.close(fig)
    save(HERE/'results/legacy_xai_audit.json',dict(track='legacy',posthoc_diagnostic_only=True,
        explained_models=20,query_rows_per_model=64,source_background_rows=64,seeds=config['seeds'],
        fixed_external_queries_across_seeds=True,source_training_background_only=True,
        local_additivity_max_residual=max(residuals),probability_scale=True,
        external_labels_used_only_for_error_reporting=True,new_common_flow_result=False,
        frozen_manifest_sha256=digest(frozen),code_sha256=digest(__file__),
        limitations='Legacy proxy semantics remain mismatched. Correlated and derived features permit off-manifold SHAP/PDP perturbations. Small fixed samples and five seeds do not establish causal explanations or deployment-wide explanation quality. Model choices use source calibration, but these analyses follow inspection of the historical final results.'))
    log('LEGACY XAI COMPLETE: external SHAP, descriptive PDP, full subtype errors and fixed-query seed stability; not common-flow evidence')


if __name__=='__main__':
    explain()
