from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from protocol import group_split, summaries
from data_revision import ROOT,load_pool,prepare
from run_revision import CONFIG,DATASETS,partition,Transform,fit,importance_weights,explain

OUT=ROOT/'revision'/'results'


def main():
    choices=pd.read_csv(OUT/'selected_cross_runs.csv').drop_duplicates(['source','target','seed'])
    pools={name:load_pool(name) for name in DATASETS}
    rows=[]
    for seed in CONFIG['seeds']:
        datasets={name:prepare(name,pools[name],CONFIG['sample_rows_per_dataset'],seed) for name in DATASETS}
        for _,choice in choices.query('seed==@seed').iterrows():
            source,target=datasets[choice.source],datasets[choice.target]
            train,cal,_=partition(source,seed)
            adapt,evaluation=group_split(target['x'],target['y'],.5,seed+2,require_both=False)
            models={(choice.model,choice.variant),('lr','baseline')}
            for kind,variant in sorted(models):
                transform=Transform(variant).fit(source['x'].iloc[train],target['x'].iloc[adapt])
                a,b=[transform.apply(source['x'].iloc[idx],source=True) for idx in [train,cal]]
                u,e=[transform.apply(target['x'].iloc[idx]) for idx in [adapt,evaluation]]
                w=importance_weights(a,u,seed) if variant.endswith('iw') or variant=='iw' else None
                model=fit(kind,a,source['y'][train],b,source['y'][cal],seed,target=u if kind=='dann' else None,weights=w)
                base=dict(source=choice.source,target=choice.target,seed=seed,model=kind,variant=variant,
                    selected=(kind==choice.model and variant==choice.variant))
                explain(model,transform,a,b,e,base,rows)
            pd.DataFrame(rows).to_csv(OUT/'external_shap_complete.csv',index=False)
            print(f'external SHAP {choice.source}->{choice.target} {seed}',flush=True)
    d=pd.DataFrame(rows)
    comparisons=[]
    for keys,g in d.groupby(['source','target','seed','model','variant']):
        a=g.query("domain=='source'").set_index('feature').mean_abs_shap
        b=g.query("domain=='target'").set_index('feature').mean_abs_shap
        intersection=a.index.intersection(b.index)
        topa=set(a.nlargest(5).index);topb=set(b.nlargest(5).index)
        comparisons.append(dict(zip(['source','target','seed','model','variant'],keys),
            top5_jaccard=len(topa&topb)/len(topa|topb),
            rank_spearman=float(spearmanr(a.loc[intersection],b.loc[intersection]).statistic)))
    pd.DataFrame(comparisons).to_csv(OUT/'shap_source_target_stability.csv',index=False)
    stability=[]
    for keys,g in d.groupby(['source','target','domain','model','variant']):
        for s1 in sorted(g.seed.unique()):
            for s2 in sorted(g.seed.unique()):
                if s2<=s1:
                    continue
                a=g.query('seed==@s1').set_index('feature').mean_abs_shap
                b=g.query('seed==@s2').set_index('feature').mean_abs_shap
                topa=set(a.nlargest(5).index);topb=set(b.nlargest(5).index)
                stability.append(dict(zip(['source','target','domain','model','variant'],keys),
                    seed_a=s1,seed_b=s2,top5_jaccard=len(topa&topb)/len(topa|topb)))
    pd.DataFrame(stability).to_csv(OUT/'shap_between_seed_stability.csv',index=False)


if __name__=='__main__':
    main()
