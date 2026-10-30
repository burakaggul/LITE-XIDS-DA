"""Associate raw common flows with source labels without deriving labels from predictors."""
from pathlib import Path
import argparse
import hashlib
import json
import time
import duckdb
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common import HERE,ROOT,digest,log,save
from features import FEATURES

PROTOCOL={'tcp':6,'udp':17,'icmp':1,'icmp6':58,'ipv6-icmp':58}


class LabelCoverageError(RuntimeError):
    pass


def canonical_keys(frame,raw=False):
    protocol=frame['proto'].astype(str).str.lower().map(PROTOCOL) if raw else pd.to_numeric(frame['protocol_number'])
    src=pd.to_numeric(frame['src_port'],errors='coerce').fillna(-1).astype('int64')
    dst=pd.to_numeric(frame['dst_port'],errors='coerce').fillna(-1).astype('int64')
    icmp=protocol.isin([1,58]);src=src.mask(icmp,0);dst=dst.mask(icmp,0)
    a=frame.src_ip.astype(str).str.strip().str.lower()+':'+src.astype(str)
    b=frame.dst_ip.astype(str).str.strip().str.lower()+':'+dst.astype(str)
    return pd.Series(np.where(a<=b,a+'|'+b,b+'|'+a),index=frame.index)+'|'+protocol.fillna(-1).astype('int64').astype(str)


def observational_identity(frame):
    a=frame.src_ip.astype(str)+':'+frame.src_port.astype(str)
    b=frame.dst_ip.astype(str)+':'+frame.dst_port.astype(str)
    forward=a<=b
    identity=frame[FEATURES+['first_ms','last_ms']].copy()
    identity['native_endpoints']=np.where(forward,a+'|'+b,b+'|'+a)
    for field in ['packets','bytes']:
        identity['low_'+field]=np.where(forward,frame['src_'+field],frame['dst_'+field])
        identity['high_'+field]=np.where(forward,frame['dst_'+field],frame['src_'+field])
    return pd.util.hash_pandas_object(identity,index=False).astype(str)


def index_source(row):
    source=ROOT/row['local_path'].replace('\\','/')
    out=HERE/'label_index'/(Path(row['path']).stem+'.parquet')
    meta=out.with_suffix('.json')
    if out.exists() and meta.exists() and json.loads(meta.read_text())['sha256']==digest(source):
        return
    out.parent.mkdir(exist_ok=True)
    cols=['ts','src_ip','src_port','dst_ip','dst_port','proto','duration','label','type']
    writer=None;count=0;bad=0
    log('LABEL INDEX START '+row['path'])
    try:
        for chunk in pd.read_csv(source,usecols=cols,chunksize=200000,low_memory=False,encoding='utf-8-sig'):
            y=pd.to_numeric(chunk.label,errors='raise')
            if not set(y.unique())<={0,1}:
                raise ValueError('Unknown source binary labels.')
            start=pd.to_numeric(chunk.ts,errors='coerce')
            duration=pd.to_numeric(chunk.duration,errors='coerce')
            valid=start.notna() & duration.notna() & duration.ge(0)
            bad+=int((~valid).sum())
            selected=pd.DataFrame(dict(key=canonical_keys(chunk,True),start_s=start,end_s=start+duration,
                label=y.astype('int64'),attack=chunk['type'].astype(str)))
            selected=selected.loc[valid]
            table=pa.Table.from_pandas(selected,preserve_index=False)
            if writer is None:
                writer=pq.ParquetWriter(str(out)+'.tmp',table.schema,compression='zstd')
            writer.write_table(table);count+=len(selected)
    finally:
        if writer is not None:
            writer.close()
    Path(str(out)+'.tmp').replace(out)
    save(meta,dict(input=row['path'],sha256=digest(source),indexed_rows=count,invalid_time_rows=bad))
    log(f'LABEL INDEX PASS {row["path"]}: {count:,} original label intervals, {bad:,} invalid-time rows excluded')


def join_labels(flows,files,tolerance,schema='v1'):
    con=duckdb.connect()
    con.execute("SET threads=4")
    con.execute("SET memory_limit='8GB'")
    con.execute("SET temp_directory='"+str(HERE/'cache/duckdb').replace('\\','/').replace("'","''")+"'")
    meta=flows[['flow_uid','first_ms','last_ms']].copy()
    meta['key']=canonical_keys(flows).to_numpy()
    if schema=='v3':
        from label_schema_v3 import normalize_keys
        meta['key']=normalize_keys(meta['key'])
    elif schema!='v1':
        raise ValueError('Unknown annotation tuple schema.')
    meta=meta.drop_duplicates('flow_uid')
    con.register('f',meta)
    query='''WITH labels AS (SELECT * FROM read_parquet(?)),
        intervals AS (
          SELECT f.flow_uid,l.label,l.attack,f.first_ms/1000.0 AS fstart,
            f.last_ms/1000.0 AS fend,
            greatest(l.start_s-?,f.first_ms/1000.0) AS lo,
            least(l.end_s+?,f.last_ms/1000.0) AS hi
          FROM f JOIN labels l ON f.key=l.key
            AND l.start_s<=f.last_ms/1000.0+?
            AND l.end_s>=f.first_ms/1000.0-?
        ), scanned AS (
          SELECT *,max(hi) OVER (PARTITION BY flow_uid ORDER BY lo,hi
            ROWS BETWEEN UNBOUNDED PRECEDING AND 1 PRECEDING) AS previous_end
          FROM intervals
        ), supported AS (
          SELECT flow_uid,count(label) AS label_support,min(label) AS min_label,
            max(label) AS max_label,string_agg(DISTINCT attack,';' ORDER BY attack) AS matched_types,
            min(lo)<=min(fstart) AND max(hi)>=max(fend)
              AND count_if(previous_end IS NOT NULL AND lo>previous_end)=0 AS time_supported
          FROM scanned GROUP BY flow_uid
        )
        SELECT f.flow_uid,coalesce(s.label_support,0) AS label_support,s.min_label,s.max_label,
          s.matched_types,coalesce(s.time_supported,FALSE) AS time_supported
        FROM f LEFT JOIN supported s ON f.flow_uid=s.flow_uid'''
    answer=con.execute(query,[list(map(str,files)),tolerance,tolerance,tolerance,tolerance]).df()
    con.close()
    answer['joined_label']=answer.min_label.where((answer.min_label==answer.max_label)&answer.time_supported)
    answer['label_match_status']=np.select([answer.label_support.eq(0).to_numpy(dtype=bool),
        (answer.min_label.ne(answer.max_label)&answer.min_label.notna()).fillna(False).to_numpy(dtype=bool),
        (~answer.time_supported).to_numpy(dtype=bool)],
        ['no_tuple_time_support','conflicting_binary_labels','unsupported_time_gap'],default='consistent_supported_union')
    return answer


def prepare(official_events=False,label_schema='v1',packet_witnesses=False):
    config=json.loads((HERE/'config.json').read_text())
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    frames=[]
    for row in plan['files']:
        if row['kind']!='pcap':
            continue
        name=hashlib.sha256(row['capture_id'].encode()).hexdigest()[:20]+'.csv.gz'
        path=HERE/'frames'/row['dataset']/name
        if not path.exists():
            raise RuntimeError('Missing extracted capture: '+row['capture_id'])
        frames.append(pd.read_csv(path,low_memory=False))
    joined=pd.concat(frames,ignore_index=True)
    ton=joined[joined.dataset=='ton_iot'].copy()
    files=[HERE/'label_index'/(Path(r['path']).stem+'.parquet') for r in plan['files'] if r['kind']=='labels']
    if any(not p.exists() for p in files):
        raise RuntimeError('Not all original-label tables are indexed.')
    official_manifest=None
    if official_events:
        official_manifest=HERE/'manifests/official_event_sources.json'
        supplemental=json.loads(official_manifest.read_text())
        if not supplemental['complete'] or len(supplemental['files'])!=18:
            raise RuntimeError('Official supplementary annotations are incomplete.')
        for row in supplemental['files']:
            path=HERE/row['index_path']
            if digest(path)!=row['index_sha256'] or digest(ROOT/row['raw_path'])!=row['raw_sha256']:
                raise RuntimeError('An official event annotation file changed.')
            files.append(path)
        log('LABEL JOIN AMENDMENT: 18 author-hosted point-event tables added; exact timestamps only, no event duration inferred')
    schema_manifest=None
    if label_schema=='v3':
        schema_manifest=HERE/'manifests/label_schema_v3.json'
        normalized=json.loads(schema_manifest.read_text())
        if not normalized['complete']:
            raise RuntimeError('Normalized annotation indexes are incomplete.')
        replacement={row['input']:row for row in normalized['files']}
        normalized_files=[]
        for path in files:
            row=replacement[str(path.relative_to(HERE)).replace('\\','/')]
            output=HERE/row['output']
            if digest(path)!=row['input_sha256'] or digest(output)!=row['output_sha256']:
                raise RuntimeError('A normalized annotation index changed.')
            normalized_files.append(output)
        files=normalized_files
        log('LABEL TUPLE SCHEMA V3: literal IP normalization and address-family-aware ICMP; labels and timestamps retained')
    log('LABEL JOIN START: exact tuple + full union time support + unanimous overlapping binary labels; no filename labels for TON')
    answer=join_labels(ton,files,config['label_clock_tolerance_seconds'],schema=label_schema)
    witness_manifest=None;witness_recovered=0
    if packet_witnesses:
        if label_schema!='v3':
            raise RuntimeError('Packet certificates require normalized annotation schema v3.')
        witness_manifest=HERE/'results/all_packet_witnesses_v3.json'
        witness=json.loads(witness_manifest.read_text())
        if not witness['complete']:
            raise RuntimeError('Not all packet-certificate captures were checked.')
        certificates=[]
        for row in witness['captures']:
            path=HERE/row['csv']
            if digest(path)!=row['csv_sha256']:
                raise AssertionError('A packet certificate changed.')
            certificates.append(pd.read_csv(path))
        certified=pd.concat(certificates,ignore_index=True)
        certified=certified[certified.certified].set_index('flow_uid')
        if certified.index.duplicated().any():
            raise AssertionError('Duplicate flow certificates.')
        mask=answer.flow_uid.isin(certified.index)
        labels=answer.loc[mask,'flow_uid'].map(certified.witnessed_label).astype(int)
        original=answer.loc[mask]
        if original.joined_label.notna().any() or not original.label_match_status.eq('unsupported_time_gap').all() or not original.min_label.eq(labels).all() or not original.max_label.eq(labels).all():
            raise AssertionError('A packet certificate does not match its original unanimous class support.')
        answer.loc[mask,'joined_label']=labels
        answer.loc[mask,'label_match_status']='packet_witness_supported'
        witness_recovered=int(mask.sum())
        log(f'PACKET CERTIFICATES APPLIED: {witness_recovered:,} flows; native packet counts, bytes, timestamps and fingerprint independently matched')
    ton=ton.drop(columns=['label','attack']).merge(answer,on='flow_uid',how='left',validate='many_to_one')
    ton['label']=ton.joined_label
    ton['attack']=ton.matched_types.fillna('unknown')
    ton['label_provenance']='Original Zeek intervals'+(' plus official timestamped point events (no duration inferred)' if official_events else '')+': same tuple; unanimous overlapping binary labels; interval union covers complete flow time; fixed 1-second timestamp tolerance'
    audit=ton.groupby(['capture_id','role']).agg(total=('flow_uid','size'),labeled=('label','count'),
        ambiguous=('joined_label',lambda s:0)).reset_index()
    audit['coverage']=audit.labeled/audit.total
    audit['ambiguous']=audit.capture_id.map(ton.assign(amb=(ton.min_label!=ton.max_label)&ton.min_label.notna()).groupby('capture_id').amb.sum())
    audit.to_csv(HERE/'results/ton_label_coverage.csv',index=False)
    overall=float(ton.label.notna().mean())
    quality=overall>=config['required_ton_label_coverage']
    log(f'LABEL JOIN COVERAGE: {overall:.4%}; binary conflicts/unknowns not guessed; quality gate={quality}')
    (HERE/'prepared').mkdir(exist_ok=True)
    all_data=pd.concat([joined[joined.dataset=='cic_iot_2023'],ton],ignore_index=True)
    unknown=all_data[all_data.label.isna()]
    unknown.to_parquet(HERE/'prepared/unresolved_labels.parquet',index=False)
    all_data=all_data[all_data.label.notna()].copy();all_data['label']=all_data.label.astype(int)
    # An observational fingerprint is not a proof of byte-identical raw packets.
    all_data['raw_identity']=observational_identity(all_data)
    overlap=all_data.groupby(['dataset','raw_identity']).agg(rows=('label','size'),
        label_values=('label','nunique'),roles=('role','nunique'),captures=('capture_id','nunique')).reset_index()
    overlap[(overlap.roles>1)|(overlap.label_values>1)].to_csv(HERE/'results/observational_overlap_audit.csv',index=False)
    counts=[]
    for dataset,g in all_data.groupby('dataset'):
        seen=set()
        for role in ['final','calibration','train']:
            part=g[g.role==role].copy()
            duplicate=part.raw_identity.isin(seen)
            removed=int(duplicate.sum());part=part[~duplicate]
            seen.update(part.raw_identity)
            directory=HERE/'prepared'/dataset;directory.mkdir(exist_ok=True)
            part.to_parquet(directory/(role+'.parquet'),index=False)
            counts.append(dict(dataset=dataset,role=role,rows=len(part),benign=int((part.label==0).sum()),
                attack=int((part.label==1).sum()),capture_files=part.capture_id.nunique(),observational_overlap_rows_excluded=removed,
                projected_duplicate_rows=int(part[FEATURES].duplicated().sum())))
    pd.DataFrame(counts).to_csv(HERE/'results/common_split_counts.csv',index=False)
    save(HERE/'manifests/common_dataset.json',dict(quality_gate_passed=quality,ton_label_coverage=overall,
        captures=len(frames),source_plan_sha256=digest(HERE/'manifests/capture_plan.json'),
        comparison_alignment=plan['comparison_alignment'],label_unit_cic='capture category inherited, not independently verified per flow',
        label_unit_ton='consistent-label interval union covers complete flow time against original-schema Zeek records; any overlapping binary conflict is excluded',
        unknown_labels_excluded=len(unknown),test_performance_inspected=False,partitions=counts,
        official_event_sources_sha256=digest(official_manifest) if official_manifest else None,
        annotation_amendment_sha256=digest(HERE/'manifests/official_annotation_amendment.json') if official_events else None,
        label_schema_version=label_schema,label_schema_manifest_sha256=digest(schema_manifest) if schema_manifest else None,
        packet_witness_manifest_sha256=digest(witness_manifest) if witness_manifest else None,
        packet_witness_recovered=witness_recovered,
        duplicate_identity_limitation='Orientation-invariant native endpoints, times, directional counters and all common statistics; packet-byte identity and acquisition-session independence are not established'))
    if not quality:
        raise LabelCoverageError('TON label coverage below predeclared quality requirement; do not call common-flow evaluation confirmatory.')
    log('PASS common data prepared; final labels remain outside model-selection APIs')


def main():
    p=argparse.ArgumentParser();p.add_argument('--stage',choices=['index','prepare'],required=True);p.add_argument('--watch',action='store_true')
    p.add_argument('--official-events',action='store_true')
    p.add_argument('--label-schema',choices=['v1','v3'],default='v1')
    p.add_argument('--packet-witnesses',action='store_true')
    a=p.parse_args()
    if a.stage=='prepare':
        prepare(a.official_events,a.label_schema,a.packet_witnesses);return
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    pending=[r for r in plan['files'] if r['kind']=='labels']
    while pending:
        ready=[r for r in pending if (ROOT/r['local_path'].replace('\\','/')).exists()]
        if not ready:
            if not a.watch:
                break
            time.sleep(5);continue
        for row in ready:
            index_source(row);pending.remove(row)
    log('LABEL INDEX STATUS: '+str(len(pending))+' files pending')


if __name__=='__main__':
    try:
        main()
    except LabelCoverageError as error:
        log('DECLARED LABEL QUALITY GATE FAILED: '+str(error))
        raise SystemExit(3)
