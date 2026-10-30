"""Lossless endpoint normalization and Zeek ICMP address-family disambiguation."""
import argparse
import ipaddress
import json
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common import HERE,digest,log,save

VERSION='zeek-ip-family-v3'
ADDRESS_CACHE={}
MISSING_ADDRESSES={'0','-','nan',''}


def normalize_address(value):
    if value not in ADDRESS_CACHE:
        ADDRESS_CACHE[value]=value if value in MISSING_ADDRESSES else ipaddress.ip_address(value).compressed.lower()
    return ADDRESS_CACHE[value]


def normalize_keys(keys):
    parts=keys.astype(str).str.split('|',expand=True,regex=False)
    if parts.shape[1]!=3:
        raise ValueError('A stored key is not a three-component canonical tuple.')
    a=parts[0].str.rsplit(':',n=1,expand=True)
    b=parts[1].str.rsplit(':',n=1,expand=True)
    left=a[0].map(normalize_address);right=b[0].map(normalize_address)
    protocol=pd.to_numeric(parts[2],errors='raise').astype('int64')
    v6a=left.str.contains(':',regex=False);v6b=right.str.contains(':',regex=False)
    valid=~left.isin(MISSING_ADDRESSES)&~right.isin(MISSING_ADDRESSES)
    if ((v6a!=v6b)&valid).any():
        raise ValueError('Mixed address families in a single stored transport tuple.')
    protocol=protocol.mask(protocol.eq(1)&v6a&v6b,58)
    first=left+':'+a[1];second=right+':'+b[1]
    return pd.Series(np.where(first<=second,first+'|'+second,second+'|'+first),index=keys.index)+'|'+protocol.astype(str)


def convert_index(source,destination):
    source=Path(source);destination=Path(destination);meta=destination.with_suffix('.json')
    source_sha=digest(source)
    if meta.exists() and destination.exists():
        old=json.loads(meta.read_text())
        if old['input_sha256']==source_sha and old['code_sha256']==digest(__file__) and old['output_sha256']==digest(destination):
            return old
        raise RuntimeError('A versioned label index changed; preserve it before a new amendment.')
    destination.parent.mkdir(parents=True,exist_ok=True)
    writer=None;rows=0;changed=0;icmp6=0
    try:
        for batch in pq.ParquetFile(source).iter_batches(batch_size=200000):
            original=pa.Table.from_batches([batch])
            keys=original.column('key').to_pandas()
            corrected=normalize_keys(keys)
            changed+=int(keys.ne(corrected).sum())
            icmp6+=int((keys.str.endswith('|1')&corrected.str.endswith('|58')).sum())
            result=original.set_column(original.schema.get_field_index('key'),'key',pa.array(corrected,type=pa.string()))
            if not result.drop(['key']).equals(original.drop(['key'])):
                raise AssertionError('Label, timestamp or type changed during schema normalization.')
            if writer is None:
                writer=pq.ParquetWriter(str(destination)+'.tmp',result.schema,compression='zstd')
            writer.write_table(result);rows+=len(result)
    finally:
        if writer is not None:
            writer.close()
    Path(str(destination)+'.tmp').replace(destination)
    record=dict(input=str(source.relative_to(HERE)).replace('\\','/'),
        output=str(destination.relative_to(HERE)).replace('\\','/'),input_sha256=source_sha,
        output_sha256=digest(destination),rows=rows,keys_normalized=changed,ipv6_icmp_protocol_corrected=icmp6,
        labels_timestamps_types_unchanged=True,version=VERSION,code_sha256=digest(__file__))
    save(meta,record)
    log(f'LABEL SCHEMA V3 {source.name}: {rows:,} rows; {changed:,} normalized keys; {icmp6:,} IPv6 ICMP corrections; labels/times unchanged')
    return record


def build():
    if any((HERE/'models/common').rglob('frozen.json')):
        raise RuntimeError('Schema amendments must precede common model fitting.')
    original_manifest=HERE/'manifests/common_dataset.json'
    amendment=HERE/'manifests/annotation_schema_amendment_v3.json'
    if not amendment.exists():
        save(amendment,dict(version=VERSION,prior_data_manifest_sha256=digest(original_manifest),
            declared_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat(),
            correction='Normalize literal IP representations; Zeek proto=icmp plus IPv6 endpoints means IP protocol 58, not IPv4 protocol 1.',
            reference='https://docs.zeek.org/en/lts/reference/logs/conn.html',
            no_label_inference=True,label_values_and_timestamps_preserved=True,
            quality_threshold_unchanged=.95,capture_roles_unchanged=True,
            common_model_performance_observed=False,code_sha256=digest(__file__)))
    files=[]
    for source_directory,destination_directory in [('label_index','label_index_v3'),('official_event_index','official_event_index_v3')]:
        for source in sorted((HERE/source_directory).glob('*.parquet')):
            files.append(convert_index(source,HERE/destination_directory/source.name))
    if len(files)!=41:
        raise AssertionError('All 23 interval tables and 18 official point-event tables are required.')
    save(HERE/'manifests/label_schema_v3.json',dict(version=VERSION,files=files,complete=True,
        amendment_sha256=digest(amendment),code_sha256=digest(__file__)))
    log('LABEL SCHEMA V3 COMPLETE: 41 versioned indexes; no class value or event time changed')


if __name__=='__main__':
    argparse.ArgumentParser().parse_args()
    build()
