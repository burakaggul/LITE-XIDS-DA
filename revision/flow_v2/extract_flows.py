"""Offline NFStream only: full capture pass, bounded deterministic sampled output."""
import argparse
import gzip
import csv
import json
import time
import platform
from pathlib import Path
from common import HERE,ROOT,digest,log,save
from features import FEATURES,common_features,flow_fingerprint,HashReservoir
from chronology import ensure,VERSION
import psutil

INITIAL_AFFINITY=psutil.Process().cpu_affinity()
EXTRACTION_CODE_SHA=digest(Path(__file__))
CHRONOLOGY_CODE_SHA=digest(HERE/'chronology.py')


def output_path(row):
    name=__import__('hashlib').sha256(row['capture_id'].encode()).hexdigest()[:20]+'.csv.gz'
    return HERE/'frames'/row['dataset']/name


def extract_capture(row,config):
    from nfstream import NFStreamer
    source=ROOT/row['local_path'].replace('\\','/')
    if not source.is_file():
        raise FileNotFoundError(source)
    chronology=ensure(row)
    normalized_source=Path(chronology['normalized_path'])
    output=output_path(row)
    metadata=output.with_suffix('.json')
    if output.exists() and metadata.exists():
        old=json.loads(metadata.read_text())
        if old['config_sha256']!=digest(HERE/'config.json') or old['source_sha256']!=digest(source) or old.get('chronology_version')!=VERSION or old.get('normalized_source_sha256')!=chronology['normalized_sha256'] or old.get('extraction_code_sha256')!=EXTRACTION_CODE_SHA:
            raise RuntimeError('An extracted capture does not match the locked source/config.')
        return old
    source_sha=digest(source)
    if row.get('sha256') and row['sha256']!=source_sha:
        raise RuntimeError('Raw source checksum mismatch.')
    e=config['extractor']
    kwargs=dict(idle_timeout=e['idle_timeout_seconds'],active_timeout=e['active_timeout_seconds'],
        accounting_mode=e['accounting_mode'],decode_tunnels=e['decode_tunnels'],n_dissections=e['n_dissections'],
        statistical_analysis=e['statistical_analysis'],n_meters=e['n_meters'],
        bpf_filter='ip or ip6',snapshot_length=65535)
    psutil.Process().cpu_affinity(INITIAL_AFFINITY)
    stream=NFStreamer(source=str(normalized_source),**kwargs)
    reservoir=HashReservoir(config['reservoir_flows_per_capture'])
    first,last=None,None
    start=time.perf_counter();last_progress=start
    log('EXTRACT START '+row['capture_id'])
    for flow in stream:
        first=min(first,flow.bidirectional_first_seen_ms) if first is not None else flow.bidirectional_first_seen_ms
        last=max(last,flow.bidirectional_last_seen_ms) if last is not None else flow.bidirectional_last_seen_ms
        uid=flow_fingerprint(flow,row['capture_id'])
        def record():
            return dict(flow_uid=uid,dataset=row['dataset'],capture_id=row['capture_id'],role=row['role'],
                src_ip=flow.src_ip,dst_ip=flow.dst_ip,src_port=flow.src_port,dst_port=flow.dst_port,
                first_ms=flow.bidirectional_first_seen_ms,last_ms=flow.bidirectional_last_seen_ms,
                src_packets=flow.src2dst_packets,dst_packets=flow.dst2src_packets,
                src_bytes=flow.src2dst_bytes,dst_bytes=flow.dst2src_bytes,
                label=(int(row['stratum']=='attack') if row['dataset']=='cic_iot_2023' else ''),
                attack=(Path(row['path']).parent.name if row['dataset']=='cic_iot_2023' else ''),
                label_provenance=row['label_provenance'],**common_features(flow))
        reservoir.offer(uid,record)
        if time.perf_counter()-last_progress>15:
            log(f'EXTRACT {row["capture_id"]}: {reservoir.seen:,} flows observed, {len(reservoir.heap):,} retained')
            last_progress=time.perf_counter()
    rows=reservoir.records()
    if not rows:
        raise RuntimeError('No IP flows extracted; do not replace with synthetic rows.')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    with gzip.open(temporary,'wt',encoding='utf-8',newline='') as handle:
        writer=csv.DictWriter(handle,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    temporary.replace(output)
    result=dict(capture_id=row['capture_id'],dataset=row['dataset'],role=row['role'],
        source_sha256=source_sha,config_sha256=digest(HERE/'config.json'),n_flows_observed=reservoir.seen,
        n_flows_saved=len(rows),sample_duplicate_fingerprints=len(rows)-len({r['flow_uid'] for r in rows}),
        sampling='order-independent hash-priority bounded sample; identical observational fingerprints share priority',
        first_ms=first,last_ms=last,seconds=time.perf_counter()-start,kwargs=kwargs,
        output=str(output.relative_to(ROOT)),output_sha256=digest(output),extractor_version=__import__('nfstream').__version__,
        python=platform.python_version(),platform=platform.platform())
    result.update(chronology_version=VERSION,normalized_source_sha256=chronology['normalized_sha256'],
        chronology_sorted_copy=chronology['sorted_copy'],effective_n_meters=stream.n_meters,
        extraction_code_sha256=EXTRACTION_CODE_SHA,chronology_code_sha256=CHRONOLOGY_CODE_SHA)
    save(metadata,result)
    log(f'EXTRACT PASS {row["capture_id"]}: {reservoir.seen:,} complete-pass flows, {len(rows):,} saved in {result["seconds"]:.1f}s')
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--watch',action='store_true');p.add_argument('--one',action='store_true')
    args=p.parse_args()
    config=json.loads((HERE/'config.json').read_text())
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    if plan['config_sha256']!=digest(HERE/'config.json'):
        raise RuntimeError('Capture plan/config mismatch.')
    pending={r['capture_id']:r for r in plan['files'] if r['kind']=='pcap'}
    completed=[]
    while pending:
        ready=[r for r in pending.values() if (ROOT/r['local_path'].replace('\\','/')).is_file()]
        if not ready:
            if not args.watch:
                break
            time.sleep(5);continue
        for row in sorted(ready,key=lambda r:r['bytes']):
            completed.append(extract_capture(row,config))
            del pending[row['capture_id']]
            save(HERE/'results/extraction.json',dict(captures=completed,pending=list(pending),complete=not pending))
            if args.one:
                return
    log(f'EXTRACTION STATUS: {len(completed)} completed, {len(pending)} pending')


if __name__=='__main__':
    main()
