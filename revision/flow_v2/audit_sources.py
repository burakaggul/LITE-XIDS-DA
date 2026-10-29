"""Independent pinned-object integrity and capture/label/split audit."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import requests
import numpy as np
import pandas as pd
from common import HERE,ROOT,digest,log,save
from features import FEATURES


def pinned_metadata(plan):
    cache=HERE/'manifests/pinned_object_integrity.json'
    if cache.exists():
        return json.loads(cache.read_text())
    repositories=sorted({(r['repository'],r['revision']) for r in plan['files']})
    metadata={}
    for repository,revision in repositories:
        url='https://huggingface.co/api/datasets/'+repository+'/tree/'+revision+'?recursive=true&limit=1000'
        while url:
            response=requests.get(url,timeout=60);response.raise_for_status()
            for row in response.json():
                if row['type']=='file':
                    metadata[repository+':'+row['path']]=dict(bytes=row['size'],git_oid=row['oid'],
                        lfs_sha256=row.get('lfs',{}).get('oid'),revision=revision)
            url=response.links.get('next',{}).get('url')
    save(cache,metadata)
    return metadata


def check_sources(watch=False):
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    metadata=pinned_metadata(plan);pending=list(plan['files']);records=[]
    while pending:
        ready=[r for r in pending if (ROOT/r['local_path'].replace('\\','/')).exists()]
        for row in ready:
            path=ROOT/row['local_path'].replace('\\','/');meta=metadata[row['repository']+':'+row['path']]
            if path.stat().st_size!=meta['bytes']:
                raise AssertionError('Pinned source size mismatch.')
            sha=digest(path);method='LFS SHA256'
            if meta['lfs_sha256']:
                if sha!=meta['lfs_sha256']:
                    raise AssertionError('Pinned source SHA256 mismatch.')
            else:
                h=hashlib.sha1(('blob '+str(path.stat().st_size)+'\0').encode())
                with path.open('rb') as stream:
                    for chunk in iter(lambda:stream.read(2**20),b''):
                        h.update(chunk)
                if h.hexdigest()!=meta['git_oid']:
                    raise AssertionError('Pinned source Git object mismatch.')
                method='Git blob SHA1 + recorded SHA256'
            records.append(dict(capture_id=row['capture_id'],bytes=path.stat().st_size,sha256=sha,
                pinned_integrity_method=method,mirror_verified_not_official_copy=True))
            pending.remove(row)
            log('INDEPENDENT SOURCE AUDIT PASS '+row['capture_id']+' ('+method+')')
        save(HERE/'results/source_integrity.json',dict(files=records,complete=not pending,pending=len(pending)))
        if not pending or not watch:
            break
        time.sleep(10)


def check_prepared():
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    rows=[];cohorts=[];collision=[]
    for dataset in ['cic_iot_2023','ton_iot']:
        frames={role:pd.read_parquet(HERE/'prepared'/dataset/(role+'.parquet')) for role in ['train','calibration','final']}
        whole=pd.concat(list(frames.values()),ignore_index=True)
        np.testing.assert_allclose(whole.packet_total,whole.src_packets+whole.dst_packets,rtol=0,atol=0)
        np.testing.assert_allclose(whole.bytes_total,whole.src_bytes+whole.dst_bytes,rtol=0,atol=0)
        np.testing.assert_allclose(whole.duration_s,(whole.last_ms-whole.first_ms)/1000.,rtol=1e-10,atol=1e-10)
        np.testing.assert_allclose(whole.bytes_mean,whole.bytes_total/whole.packet_total,rtol=1e-10,atol=1e-10)
        if not ((whole.packet_total>=1)&(whole.duration_s>=0)&(whole.bytes_min<=whole.bytes_mean+1e-8)&
            (whole.bytes_mean<=whole.bytes_max+1e-8)).all():
            raise AssertionError('Impossible flow counts/durations/packet-size statistics.')
        duration=whole.duration_s.to_numpy();rate=whole.byte_rate_s.to_numpy()
        np.testing.assert_allclose(rate[duration>0],(whole.bytes_total/whole.duration_s)[duration>0],rtol=1e-10,atol=1e-10)
        if not np.isnan(rate[duration==0]).all() or not whole.loc[whole.packet_total==1,'iat_mean_s'].isna().all():
            raise AssertionError('Undefined rates/singleton IAT values were fabricated.')
        for column,values in [('tcp',[6]),('udp',[17]),('icmp',[1,58])]:
            np.testing.assert_array_equal(whole[column],whole.protocol_number.isin(values).astype(int))
        cohorts.append(whole)
        keys=pd.util.hash_pandas_object(whole[FEATURES],index=False)
        table=pd.crosstab(keys,whole.label)
        conflict=table.index[(table>0).sum(axis=1)>1]
        across_roles=pd.DataFrame(dict(key=keys,role=whole.role)).groupby('key').role.nunique()
        collision.append(dict(dataset=dataset,rows=len(whole),unique_common_feature_vectors=len(table),
            common_feature_duplicate_rows=len(whole)-len(table),label_conflict_keys=len(conflict),
            label_conflict_rows=int(keys.isin(conflict).sum()),feature_keys_shared_across_roles=int((across_roles>1).sum()),
            observed_majority_lookup_accuracy_ceiling=float(table.max(axis=1).sum()/len(whole))))
        for role,frame in frames.items():
            if set(frame.label.unique())!={0,1}:
                raise AssertionError('A prepared partition lacks a binary class: '+dataset+'/'+role)
            planned={r['capture_id'] for r in plan['files'] if r['dataset']==dataset and r['kind']=='pcap' and r['role']==role}
            if not set(frame.capture_id)<=planned:
                raise AssertionError('Capture role changed.')
            if frame.flow_uid.isna().any():
                raise AssertionError('Missing flow identity.')
            for other,other_frame in frames.items():
                if other==role:
                    continue
                if set(frame.capture_id)&set(other_frame.capture_id):
                    raise AssertionError('Capture overlap.')
                if set(frame.raw_identity)&set(other_frame.raw_identity):
                    raise AssertionError('Observational fingerprint overlap.')
            rows.append(dict(dataset=dataset,role=role,rows=len(frame),both_classes=True,
                captures=sorted(frame.capture_id.unique()),sha256=digest(HERE/'prepared'/dataset/(role+'.parquet'))))
    extraction=json.loads((HERE/'results/extraction.json').read_text())
    if not extraction['complete'] or len(extraction['captures'])!=44:
        raise AssertionError('Not all complete PCAPs were processed.')
    chronology=[]
    for capture in extraction['captures']:
        if capture.get('chronology_version')!='capture-chronology-v1':
            raise AssertionError('An unsorted-input feature cache was reused.')
        key=hashlib.sha256(capture['capture_id'].encode()).hexdigest()[:20]
        metadata=HERE/'cache/chronological'/capture['dataset']/(key+'.json')
        record=json.loads(metadata.read_text())
        if record['normalized_sha256']!=capture['normalized_source_sha256'] or record['source_sha256']!=capture['source_sha256']:
            raise AssertionError('Raw/normalized/extracted provenance chain mismatch.')
        if record['sorted_copy'] and record['preservation']['normalized']['packets_behind_running_max']!=0:
            raise AssertionError('A normalized capture is not ordered.')
        chronology.append(dict(capture_id=capture['capture_id'],sorted_copy=record['sorted_copy'],
            source_sha256=record['source_sha256'],normalized_sha256=record['normalized_sha256'],
            packet_bytes_modified=record['packet_bytes_modified'],timestamps_modified=record['timestamps_modified'],
            incomplete_records_excluded=record.get('incomplete_records_excluded',0),
            effective_n_meters=capture['effective_n_meters']))
    pd.DataFrame(chronology).to_csv(HERE/'results/capture_chronology_audit.csv',index=False)
    identity=json.loads((HERE/'results/source_integrity.json').read_text())
    if not identity['complete'] or len(identity['files'])!=67:
        raise AssertionError('Not all raw objects were independently checksum-verified.')
    whole=pd.concat(cohorts,ignore_index=True)
    keys=pd.util.hash_pandas_object(whole[FEATURES],index=False)
    table=pd.crosstab(keys,whole.label);conflict=table.index[(table>0).sum(axis=1)>1]
    collision.append(dict(dataset='combined_descriptive_only_not_for_training',rows=len(whole),
        unique_common_feature_vectors=len(table),common_feature_duplicate_rows=len(whole)-len(table),
        label_conflict_keys=len(conflict),label_conflict_rows=int(keys.isin(conflict).sum()),
        observed_majority_lookup_accuracy_ceiling=float(table.max(axis=1).sum()/len(whole))))
    pd.DataFrame(collision).to_csv(HERE/'results/common_feature_collision_audit.csv',index=False)
    save(HERE/'results/prepared_audit.json',dict(partitions=rows,full_capture_count=44,raw_object_count=67,
        requested_extraction_workers=4,effective_extraction_workers=sorted({c['effective_n_meters'] for c in extraction['captures']}),
        limitation='Disjoint files/fingerprints do not establish independent acquisition sessions or packet-byte identity.',
        feature_collisions='Distinct raw-flow observations with identical 12-feature vectors are retained. Collision/label-conflict diagnostics are descriptive only; no final labels enter model selection. Observed majority-lookup ceilings are not achievable out-of-sample performance guarantees.',
        physical_invariants_verified='Directional totals, duration/time units, packet-size range/mean, byte rate, missing zero-duration rates/singleton IAT and protocol flags checked on all prepared observations.',
        original_submitted_pdf_sha256=digest(ROOT/'output/pdf/Manuscript.pdf') if (ROOT/'output/pdf/Manuscript.pdf').exists() else None))
    log('PREPARED PHYSICAL AUDIT PASS: 67 pinned raw objects, 44 complete-record capture passes, disjoint roles and observed identities; label coverage is a separate gate')


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['sources','prepared'],required=True)
    p.add_argument('--watch',action='store_true');a=p.parse_args()
    check_sources(a.watch) if a.stage=='sources' else check_prepared()
