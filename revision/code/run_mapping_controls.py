from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from protocol import group_split,hashes,metrics,source_threshold,summaries
from data_revision import ROOT,load_pool,prepare
from run_revision import CONFIG,VALUES,partition,Transform,fit,get_scores

OUT=ROOT/'revision'/'results'


def main():
    names=['cic_iot_2023','ton_iot']
    pools={n:load_pool(n) for n in names}
    rows=[]
    for seed in CONFIG['seeds']:
        datasets={n:prepare(n,pools[n],CONFIG['sample_rows_per_dataset'],seed) for n in names}
        for source_name,target_name in [names,names[::-1]]:
            source,target=datasets[source_name],datasets[target_name]
            for space in ['proxy12','proxy8']:
                xs,xt=source['x'].copy(),target['x'].copy()
                if space=='proxy8':
                    xs=xs.drop(columns=['bytes_min','bytes_max','rate','iat_proxy'])
                    xt=xt[xs.columns]
                for control in ['predictor_grouped','random_rows_diagnostic']:
                    if control=='predictor_grouped':
                        remain,st=group_split(xs,source['y'],.2,seed)
                        a,b=group_split(xs.iloc[remain],source['y'][remain],.15/.8,seed+1)
                        train,cal=remain[a],remain[b]
                        adapt,evaluation=group_split(xt,target['y'],.5,seed+2,require_both=False)
                    else:
                        remain,st=train_test_split(np.arange(len(xs)),test_size=.2,random_state=seed,stratify=source['y'])
                        train,cal=train_test_split(remain,test_size=.15/.8,random_state=seed+1,stratify=source['y'][remain])
                        adapt,evaluation=train_test_split(np.arange(len(xt)),test_size=.5,random_state=seed+2)
                    t=Transform('robust').fit(xs.iloc[train],xt.iloc[adapt])
                    a,b=[t.apply(xs.iloc[idx],source=True) for idx in [train,cal]]
                    e=t.apply(xt.iloc[evaluation])
                    model=fit('lr',a,source['y'][train],b,source['y'][cal],seed)
                    sc,_,_=get_scores(model,b)
                    threshold=source_threshold(source['y'][cal],sc)
                    s,p,_=get_scores(model,e)
                    rows.append(dict(source=source_name,target=target_name,seed=seed,space=space,
                        control=control,source_train_n=len(train),target_eval_n=len(evaluation),
                        source_predictor_overlap=int(np.isin(hashes(xs.iloc[cal]),hashes(xs.iloc[train])).sum()),
                        target_predictor_overlap=int(np.isin(hashes(xt.iloc[evaluation]),hashes(xt.iloc[adapt])).sum()),
                        **metrics(target['y'][evaluation],(s>=threshold).astype(int),s,p)))
            pd.DataFrame(rows).to_csv(OUT/'mapping_leakage_controls_runs.csv',index=False)
    summaries(pd.DataFrame(rows),['source','target','space','control'],VALUES).to_csv(OUT/'mapping_leakage_controls_summary.csv',index=False)


if __name__=='__main__':
    main()
