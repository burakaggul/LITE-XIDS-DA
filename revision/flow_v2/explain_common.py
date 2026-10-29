"""Post-freeze probability-scale SHAP and descriptive PDP; never used for model selection."""
import json
import joblib
import numpy as np
import pandas as pd
import torch
import shap
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.inspection import partial_dependence
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import HERE,digest,log,save
from features import FEATURES
from cost_models import probability
from experiment import common_partition


class FrozenPredictor(ClassifierMixin,BaseEstimator):
    def __init__(self,model,transform):
        self.model=model;self.transform=transform
        self.classes_=np.array([0,1]);self.n_features_in_=len(transform.columns)
        self.feature_names_in_=np.asarray(transform.columns,dtype=object)

    def fit(self,x,y):
        raise RuntimeError('Explanations must use a frozen fitted model, not refit it.')

    def predict_proba(self,x):
        frame=pd.DataFrame(np.asarray(x),columns=self.transform.columns)
        p=probability(self.model,self.transform.apply(frame))
        return np.column_stack([1-p,p])


def explain():
    complete=HERE/'results/complete_common.json'
    if not complete.exists() or not (HERE/'results/result_audit_common.json').exists():
        raise RuntimeError('Final common-flow experiment and independent result audit must be complete first.')
    frozen=HERE/'manifests/frozen_common.json';manifest=json.loads(frozen.read_text())
    for file,sha in manifest['freeze_sha256'].items():
        if digest(HERE/file)!=sha:
            raise RuntimeError('Frozen source record changed.')
    torch.set_num_threads(1)
    rows=[];examples=[];stability=[];pdp=[];residuals=[];subtypes=[]
    config=json.loads((HERE/'config.json').read_text())
    for seed in config['seeds']:
        for source in config['primary_pair']:
            directory=HERE/'models/common'/source/str(seed)
            info=json.loads((directory/'frozen.json').read_text());transform=joblib.load(directory/'transform.joblib')
            train=common_partition(source,'train',seed)
            raw=train[FEATURES].sample(n=min(64,len(train)),random_state=42)
            background=pd.DataFrame(transform.imputer.transform(raw),columns=FEATURES)
            for family in ['teacher','student_distilled']:
                choice=next(c for c in info['selections'] if c['family']==family and c['arm']=='cost_sensitive' and c['fpr_limit']==.05)
                file=directory/choice['candidate_id']
                if digest(file)!=info['artifact_sha256'][choice['candidate_id']]:
                    raise RuntimeError('Frozen explanation model checksum mismatch.')
                model=joblib.load(file);predictor=FrozenPredictor(model,transform)
                f=lambda a:predictor.predict_proba(a)[:,1]
                explainer=shap.PermutationExplainer(f,background.to_numpy(),feature_names=FEATURES,seed=seed)
                domain_importance={}
                for target in config['primary_pair']:
                    whole=common_partition(target,'final',42,cap=False)
                    if target!=source:
                        full_prediction=predictor.predict_proba(whole[FEATURES])[:,1]>=choice['threshold']
                        errors=pd.DataFrame(dict(attack=whole.attack,label=whole.label,alarm=full_prediction.astype(int)))
                        for (attack,label),g in errors.groupby(['attack','label']):
                            alarms=int(g.alarm.sum());n=len(g)
                            subtypes.append(dict(source=source,target=target,seed=seed,family=family,attack=attack,
                                label=int(label),n=n,tp=alarms if label==1 else 0,fn=n-alarms if label==1 else 0,
                                fp=alarms if label==0 else 0,tn=n-alarms if label==0 else 0,
                                attack_recall=alarms/n if label==1 else np.nan,benign_fpr=alarms/n if label==0 else np.nan,
                                source_fpr_limit=.05,benign_cost=choice['benign_cost']))
                    sample=whole.sample(n=64,random_state=42)
                    x=pd.DataFrame(transform.imputer.transform(sample[FEATURES]),columns=FEATURES)
                    values=explainer(x.to_numpy(),max_evals=2*(2*len(FEATURES)+1),silent=True)
                    prediction=f(x.to_numpy())
                    residual=np.max(np.abs(values.base_values+values.values.sum(1)-prediction))
                    if residual>1e-4:
                        raise AssertionError('SHAP local additivity check failed.')
                    residuals.append(float(residual));importance=np.abs(values.values).mean(0)
                    domain_importance[target]=importance
                    for feature,value in zip(FEATURES,importance):
                        rows.append(dict(source=source,target=target,seed=seed,family=family,feature=feature,
                            mean_abs_shap_probability=float(value),query_rows=len(x),background_rows=len(background),
                            benign_cost=choice['benign_cost'],source_fpr_limit=.05))
                    if target!=source:
                        for i,(_,row) in enumerate(sample.iterrows()):
                            ranking=np.argsort(-np.abs(values.values[i]))[:3]
                            examples.append(dict(source=source,target=target,seed=seed,family=family,flow_uid=row.flow_uid,
                                label=int(row.label),attack_probability=float(prediction[i]),threshold=choice['threshold'],
                                alarm=int(prediction[i]>=choice['threshold']),
                                top_features=';'.join(FEATURES[j] for j in ranking),
                                top_signed_contributions=';'.join(str(float(values.values[i,j])) for j in ranking)))
                    if seed==42 and target!=source:
                        for feature in ['duration_s','packet_total','bytes_mean']:
                            j=FEATURES.index(feature)
                            result=partial_dependence(predictor,x,features=[j],method='brute',kind='average',grid_resolution=12)
                            for grid,value in zip(result['grid_values'][0],result['average'][0]):
                                pdp.append(dict(source=source,target=target,family=family,feature=feature,
                                    feature_value=float(grid),mean_attack_probability=float(value),rows=len(x)))
                    log(f'XAI PASS common/{source}->{target}/seed{seed}/{family}: 64 fixed rows, probability SHAP additivity {residual:.3g}')
                target=next(n for n in config['primary_pair'] if n!=source)
                a=domain_importance[source];b=domain_importance[target]
                topa=set(np.argsort(-a)[:5]);topb=set(np.argsort(-b)[:5])
                stability.append(dict(source=source,target=target,seed=seed,family=family,
                    top5_jaccard=len(topa&topb)/len(topa|topb),rank_spearman=float(spearmanr(a,b).statistic)))
                pd.DataFrame(rows).to_csv(HERE/'results/common_shap.csv',index=False)
    data=pd.DataFrame(rows);data.to_csv(HERE/'results/common_shap.csv',index=False)
    pd.DataFrame(examples).to_csv(HERE/'results/common_error_explanations.csv',index=False)
    pd.DataFrame(subtypes).to_csv(HERE/'results/common_attack_subtype_errors.csv',index=False)
    pd.DataFrame(stability).to_csv(HERE/'results/common_shap_domain_stability.csv',index=False)
    pdp=pd.DataFrame(pdp);pdp.to_csv(HERE/'results/common_pdp.csv',index=False)
    between=[]
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
    pd.DataFrame(between).to_csv(HERE/'results/common_shap_seed_stability.csv',index=False)
    out=HERE/'figures';out.mkdir(exist_ok=True)
    for (source,target,family),group in data[data.source!=data.target].groupby(['source','target','family']):
        stats=group.groupby('feature').mean_abs_shap_probability.agg(['mean','std']).sort_values('mean')
        fig,ax=plt.subplots(figsize=(7,4.7));ax.barh(stats.index,stats['mean'],xerr=stats['std'],color='#0072B2')
        ax.set(xlabel='Mean absolute SHAP contribution (attack probability)',title=source+' -> '+target+' / '+family)
        fig.tight_layout();fig.savefig(out/('shap_'+source+'_'+family+'.pdf'));plt.close(fig)
    for (source,target,family),group in pdp.groupby(['source','target','family']):
        fig,axes=plt.subplots(1,3,figsize=(11,3.5))
        for ax,(feature,g) in zip(axes,group.groupby('feature')):
            ax.plot(g.feature_value,g.mean_attack_probability,color='#009E73');ax.set(xlabel=feature,ylabel='Mean attack probability')
        fig.suptitle('Descriptive PDP: '+source+' -> '+target+' / '+family);fig.tight_layout()
        fig.savefig(out/('pdp_'+source+'_'+family+'.pdf'));plt.close(fig)
    save(HERE/'results/common_xai_audit.json',dict(shap_version=shap.__version__,seeds=config['seeds'],
        explained_families=['teacher','student_distilled'],source_selected_policy='cost-sensitive at source empirical FPR<=0.05',
        probability_scale=True,max_local_additivity_residual=max(residuals),source_fitted_background_only=True,
        fixed_query_sample_size=64,background_size=64,final_labels_used_only_for_posthoc_errors=True,
        frozen_manifest_sha256=digest(frozen),
        limitations='Permutation SHAP and brute-force PDP can form off-manifold combinations among correlated/derived flow statistics and protocol indicators. Attributions and PDP are descriptive, not causal explanations or a guarantee of fidelity outside observed traffic. Small fixed query cohorts do not estimate deployment-wide explanation quality.'))
    log('XAI COMPLETE: external and within-source probability explanations, five-seed/domain rank stability and descriptive PDP')


if __name__=='__main__':
    explain()
