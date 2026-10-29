from __future__ import annotations
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import wilcoxon
from statsmodels.stats.multitest import multipletests
from protocol import group_split, hashes, metrics, source_threshold, summaries
from data_revision import ROOT, load_pool, prepare
from run_revision import Transform, fit, get_scores, partition, CONFIG, VALUES

OUT=ROOT/'revision'/'results'


def ul_audit():
    directory=ROOT/'data'/'raw'/'ul_ece_2025'
    mqtt=pd.read_csv(next(directory.rglob('UL-ECE-MQTT-DDoS-H-IoT2025.csv')))
    udp=pd.read_csv(next(directory.rglob('UL-ECE-UDP-DDoS-H-IoT2025.csv')))
    rows=[]
    for name,d,pred,rule in [
        ('UL-ECE-MQTT-DDoS-H-IoT2025',mqtt,(mqtt.frequency>=5).astype(int),
            'frequency >= 5; frequency = 100*total_messages_each_node/total_message'),
        ('UL-ECE-UDP-DDoS-H-IoT2025',udp,(udp.monitoring_frequency>=udp.mean_frequency).astype(int),
            'monitoring_frequency >= mean_frequency')]:
        rows.append(dict(dataset=name,rows=len(d),columns=len(d.columns),benign=int((d.outcome==0).sum()),
            attack=int((d.outcome==1).sum()),published_label_rule=rule,
            rule_label_agreement=float(np.mean(pred==d.outcome)),
            use='label-construction diagnostic, not independent attack ground truth or 12-feature transfer'))
    pd.DataFrame(rows).to_csv(OUT/'ul_label_audit.csv',index=False)


def purge(train, holdout, frame):
    keys=hashes(frame)
    return train[~np.isin(keys[train],keys[holdout])]


def grouped():
    rows=[]
    for name in ['cic_iot_2023','ton_iot','cic_iomt_2024']:
        pool=load_pool(name)
        for seed in CONFIG['seeds']:
            d=prepare(name,pool,CONFIG['sample_rows_per_dataset'],seed)
            for mode in ['random_rows_diagnostic','predictor_grouped','source_device_grouped','chronological']:
                base=dict(dataset=name,seed=seed,split_mode=mode)
                if mode=='source_device_grouped' and d['device'] is None:
                    rows.append(dict(**base,status='unavailable',reason='mirror has no capture/device identifier'))
                    continue
                if mode=='chronological' and d['timestamp'] is None:
                    rows.append(dict(**base,status='unavailable',reason='mirror has no timestamp'))
                    continue
                try:
                    if mode=='random_rows_diagnostic':
                        remain,test=train_test_split(np.arange(len(d['y'])),test_size=.2,random_state=seed,stratify=d['y'])
                        train,cal=train_test_split(remain,test_size=.15/.8,random_state=seed+1,stratify=d['y'][remain])
                    elif mode=='chronological':
                        order=np.argsort(d['timestamp'].astype('int64').to_numpy(),kind='stable')
                        if d['timestamp'].isna().any():
                            raise ValueError('Unparseable timestamps')
                        train,cal,test=np.split(order,[int(.65*len(order)),int(.8*len(order))])
                        # Purge equal timestamps across boundaries rather than imply independent chronology.
                        timestamps=d['timestamp'].astype('int64').to_numpy()
                        train=train[~np.isin(timestamps[train],timestamps[np.r_[cal,test]])]
                        cal=cal[~np.isin(timestamps[cal],timestamps[test])]
                    else:
                        group=d['device'] if mode=='source_device_grouped' else hashes(d['x'])
                        remain,test=group_split(d['x'],d['y'],.2,seed,groups=group)
                        a,b=group_split(d['x'].iloc[remain],d['y'][remain],.15/.8,seed+1,groups=group[remain])
                        train,cal=remain[a],remain[b]
                    if mode!='random_rows_diagnostic':
                        train=purge(train,np.r_[cal,test],d['x'])
                        cal=purge(cal,test,d['x'])
                    if min(len(train),len(cal),len(test))<20 or len(np.unique(d['y'][train]))<2 or len(np.unique(d['y'][cal]))<2:
                        raise ValueError('Insufficient rows/classes after honest grouping and purging')
                    transform=Transform('robust').fit(d['x'].iloc[train])
                    a,b,c=[transform.apply(d['x'].iloc[idx],source=True) for idx in [train,cal,test]]
                    for kind in ['lr','rf']:
                        model=fit(kind,a,d['y'][train],b,d['y'][cal],seed)
                        score_cal,_,_=get_scores(model,b)
                        threshold=source_threshold(d['y'][cal],score_cal)
                        s,p,_=get_scores(model,c)
                        overlap=int(np.isin(hashes(d['x'].iloc[test]),hashes(d['x'].iloc[train])).sum())
                        rows.append(dict(**base,status='completed',model=kind,train_n=len(train),cal_n=len(cal),test_n=len(test),
                            train_benign=int((d['y'][train]==0).sum()),test_benign=int((d['y'][test]==0).sum()),
                            predictor_overlap_test_rows=overlap,
                            **metrics(d['y'][test],(s>=threshold).astype(int),s,p)))
                except ValueError as error:
                    rows.append(dict(**base,status='not_estimable',reason=str(error)))
            pd.DataFrame(rows).to_csv(OUT/'split_sensitivity_runs.csv',index=False)
            print(f'split sensitivity {name} {seed}',flush=True)
    valid=pd.DataFrame(rows).query("status=='completed'")
    summaries(valid,['dataset','split_mode','model'],VALUES).to_csv(OUT/'split_sensitivity_summary.csv',index=False)


def stats():
    d=pd.read_csv(OUT/'cross_runs.csv')
    # Paired seeds; each direction and endpoint is a separate declared family of comparisons.
    rows=[]
    for (source,target),g in d.query("operating_point=='source_f1' and model=='lr'").groupby(['source','target']):
        for endpoint in ['macro_f1','fpr']:
            baseline=g.query("variant=='baseline'").set_index('seed')[endpoint]
            for variant in ['robust','iw','robust_iw','robust_coral','robust_gate','robust_gate_iw']:
                alt=g.query('variant==@variant').set_index('seed')[endpoint]
                idx=baseline.index.intersection(alt.index)
                delta=alt.loc[idx]-baseline.loc[idx]
                p=1. if np.allclose(delta,0) else float(wilcoxon(delta,zero_method='wilcox',method='exact').pvalue)
                rows.append(dict(source=source,target=target,endpoint=endpoint,comparison=variant+' minus baseline',
                    n_pairs=len(idx),mean_difference=float(delta.mean()),p_unadjusted=p))
    r=pd.DataFrame(rows)
    r['p_holm']=np.nan
    for _,g in r.groupby(['source','target','endpoint']):
        r.loc[g.index,'p_holm']=multipletests(g.p_unadjusted,method='holm')[1]
    r.to_csv(OUT/'paired_tests.csv',index=False)


if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--statistics-only',action='store_true')
    args=p.parse_args()
    if args.statistics_only:
        stats()
    else:
        ul_audit()
        grouped()
