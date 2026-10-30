"""Descriptive explanations and matched-backend CPU costs of frozen raw models."""
import io
import json
import time
import joblib
import numpy as np
import pandas as pd
import psutil
import shap
from scipy.stats import spearmanr
from sklearn.inspection import partial_dependence
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import digest,log,save
from cost_models import RowWiseQuantized,neural_bytes
from explain_common import FrozenPredictor
from features import FEATURES
from labeled_raw_experiment import RUN,verify_freeze,partition


def main():
    verify_freeze()
    if not (RUN/'results/verified.json').exists():
        raise RuntimeError('Independent raw-result verification must precede descriptive analyses.')
    config=json.loads((RUN/'config.json').read_text());torch.set_num_threads(1)
    importance=[];examples=[];domain_stability=[];pdp=[];interactions=[];cpu=[];residuals=[]
    for source in config['primary_pair']:
        target=next(n for n in config['primary_pair'] if n!=source)
        for seed in config['seeds']:
            directory=RUN/'models'/source/str(seed);info=json.loads((directory/'frozen.json').read_text())
            transform=joblib.load(directory/'transform.joblib')
            training=partition(source,'train',seed)
            background=pd.DataFrame(transform.imputer.transform(training[FEATURES].sample(n=64,random_state=42)),columns=FEATURES)
            for family in ['teacher','student_distilled']:
                choice=next(c for c in info['selections'] if c['family']==family and c['arm']=='cost_sensitive' and c['fpr_limit']==.01)
                file=directory/choice['candidate_id']
                if digest(file)!=info['artifact_sha256'][file.name]:
                    raise AssertionError('An explanation artifact changed.')
                model=joblib.load(file);predictor=FrozenPredictor(model,transform)
                predict=lambda a:predictor.predict_proba(a)[:,1]
                explainer=shap.PermutationExplainer(predict,background.to_numpy(),feature_names=FEATURES,seed=seed)
                rankings={}
                for domain in [source,target]:
                    whole=partition(domain,'final',seed,cap=False)
                    sample=whole.sample(n=64,random_state=42)
                    x=pd.DataFrame(transform.imputer.transform(sample[FEATURES]),columns=FEATURES)
                    result=explainer(x.to_numpy(),max_evals=2*(2*len(FEATURES)+1),silent=True)
                    probability=predict(x.to_numpy())
                    residual=float(np.max(np.abs(result.base_values+result.values.sum(1)-probability)))
                    if residual>1e-4:
                        raise AssertionError('Probability-scale SHAP local additivity failed.')
                    residuals.append(residual);mean=np.abs(result.values).mean(0);rankings[domain]=mean
                    for feature,value in zip(FEATURES,mean):
                        importance.append(dict(source=source,target=domain,seed=seed,family=family,feature=feature,
                            mean_abs_shap_probability=float(value),source_fpr_limit=.01,benign_cost=choice['benign_cost']))
                    if domain==target:
                        for i,row in enumerate(sample.itertuples()):
                            top=np.argsort(-np.abs(result.values[i]))[:3]
                            examples.append(dict(source=source,target=domain,seed=seed,family=family,flow_uid=row.flow_uid,
                                label=int(row.label),probability=float(probability[i]),threshold=choice['threshold'],
                                top_features=';'.join(FEATURES[j] for j in top),
                                signed_contributions=';'.join(str(float(result.values[i,j])) for j in top)))
                        if seed==42:
                            for feature in ['duration_s','packet_total','bytes_mean']:
                                effect=partial_dependence(predictor,x,[FEATURES.index(feature)],method='brute',grid_resolution=12)
                                for value,average in zip(effect['grid_values'][0],effect['average'][0]):
                                    pdp.append(dict(source=source,target=target,family=family,feature=feature,
                                        feature_value=float(value),mean_attack_probability=float(average)))
                            effect=partial_dependence(predictor,x,[(0,2)],method='brute',grid_resolution=8)
                            for i,a in enumerate(effect['grid_values'][0]):
                                for j,b in enumerate(effect['grid_values'][1]):
                                    interactions.append(dict(source=source,target=target,family=family,duration_s=float(a),
                                        packet_total=float(b),mean_attack_probability=float(effect['average'][0,i,j])))
                    log(f'RAW XAI CHECK {source}->{domain}/seed{seed}/{family}: 64 fixed queries, local-additivity residual={residual:.3g}')
                a,b=rankings[source],rankings[target];sa=set(np.argsort(-a)[:5]);sb=set(np.argsort(-b)[:5])
                domain_stability.append(dict(source=source,target=target,seed=seed,family=family,
                    top5_jaccard=len(sa&sb)/len(sa|sb),rank_spearman=float(spearmanr(a,b).statistic)))
            choice=next(c for c in info['selections'] if c['family']=='student_distilled' and c['arm']=='cost_sensitive' and c['fpr_limit']==.01)
            cost=choice['benign_cost'];input_row=partition(source,'calibration',seed).iloc[:1]
            tensor=torch.tensor(transform.apply(input_row[FEATURES]),dtype=torch.float32)
            for family in ['student_distilled','student_distilled_int8']:
                file=directory/(family+'_cost'+str(cost)+'.joblib')
                if digest(file)!=info['artifact_sha256'][file.name]:
                    raise AssertionError('Matched CPU artifact changed.')
                model=joblib.load(file);compiled=torch.jit.script(RowWiseQuantized(model).eval())
                with torch.no_grad():
                    for _ in range(100):
                        torch.softmax(compiled(tensor),1).numpy()
                    elapsed=[]
                    for _ in range(500):
                        start=time.perf_counter_ns();torch.softmax(compiled(tensor),1).numpy()
                        elapsed.append((time.perf_counter_ns()-start)/1e6)
                    if family=='student_distilled':
                        np.testing.assert_allclose(compiled(tensor).numpy(),model(tensor).numpy(),rtol=1e-5,atol=1e-5)
                buffer=io.BytesIO();torch.jit.save(compiled,buffer)
                cpu.append(dict(source=source,seed=seed,family=family,benign_cost=cost,
                    model_ms_median=float(np.median(elapsed)),model_ms_p95=float(np.quantile(elapsed,.95)),
                    state_dict_bytes=neural_bytes(model),exported_script_bytes=len(buffer.getvalue()),
                    architectural_coefficients=2978,process_rss_bytes=psutil.Process().memory_info().rss,
                    backend='TorchScript row-wise, CPU1, batch1',warmups=100,timed_calls=500,
                    scope='Desktop completed-flow model only; excludes packet capture, flow building, aggregation delay and embedded-device execution. RSS is whole process.'))
            log(f'RAW MATCHED CPU {source}/seed{seed}: float and INT8 at identical cost={cost}, backend and input')
    data=pd.DataFrame(importance);stability=[]
    for key,group in data.groupby(['source','target','family']):
        for a in config['seeds']:
            for b in config['seeds']:
                if b<=a:
                    continue
                x=group[group.seed.eq(a)].set_index('feature').mean_abs_shap_probability
                y=group[group.seed.eq(b)].set_index('feature').mean_abs_shap_probability.reindex(x.index)
                sa=set(x.nlargest(5).index);sb=set(y.nlargest(5).index)
                stability.append(dict(zip(['source','target','family'],key),seed_a=a,seed_b=b,
                    top5_jaccard=len(sa&sb)/len(sa|sb),rank_spearman=float(spearmanr(x,y).statistic)))
    for name,frame in [('shap',data),('shap_domain_stability',pd.DataFrame(domain_stability)),
        ('shap_seed_stability',pd.DataFrame(stability)),('explanation_examples',pd.DataFrame(examples)),
        ('pdp',pd.DataFrame(pdp)),('pdp_interactions',pd.DataFrame(interactions)),('matched_cpu',pd.DataFrame(cpu))]:
        frame.to_csv(RUN/'results'/(name+'.csv'),index=False)
    fig,axes=plt.subplots(2,2,figsize=(12,8))
    for ax,((source,target,family),group) in zip(axes.flat,data[data.source!=data.target].groupby(['source','target','family'])):
        values=group.groupby('feature').mean_abs_shap_probability.agg(['mean','std']).sort_values('mean')
        ax.barh(values.index,values['mean'],xerr=values['std'],color='#0072B2')
        ax.set(xlabel='Mean |SHAP| (attack probability)',title=source+' -> '+target+'\n'+family)
        ax.tick_params(labelsize=8)
    fig.tight_layout();fig.savefig(RUN/'figures/external_shap.pdf');fig.savefig(RUN/'figures/external_shap.png',dpi=180);plt.close(fig)
    save(RUN/'results/post_verified.json',dict(probability_shap_rows=len(data),explained_source_model_bundles=20,
        local_additivity_max_residual=max(residuals),domain_rank_pairs=len(domain_stability),seed_rank_pairs=len(stability),
        fixed_queries_per_domain=64,source_training_background_only=True,source_selected_policies_only=True,
        matched_backend_cpu_profiles=len(cpu),real_device_measured=False,
        interpretation='Permutation SHAP and 1D/2D brute-force PDP are descriptive; correlated/derived features can yield off-manifold perturbations. No causal interpretation or deployment-wide explanation quality is claimed.',
        code_sha256=digest(__file__)))
    log('RAW POST-ANALYSES COMPLETE: probability XAI, domain/seed stability, descriptive interactions, 20 matched CPU profiles')


if __name__=='__main__':
    main()
