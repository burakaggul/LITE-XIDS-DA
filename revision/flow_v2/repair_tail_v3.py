"""Repair explicitly warned legacy caches and re-extract affected source captures."""
import hashlib
import json
from pathlib import Path
import shutil
import pandas as pd
from common import HERE,log,save,digest
from chronology import ensure
from extract_flows import extract_capture,output_path


def run():
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    config=json.loads((HERE/'config.json').read_text())
    repairs=[]
    for row in plan['files']:
        if row['kind']!='pcap':
            continue
        key=hashlib.sha256(row['capture_id'].encode()).hexdigest()[:20]
        meta=HERE/'cache/chronological'/row['dataset']/(key+'.json')
        old=json.loads(meta.read_text())
        if 'cut short in the middle of a packet' not in old.get('tool_stderr','').lower() or old.get('structural_validation'):
            continue
        proof=ensure(row)
        archive=HERE/'archive/legacy_tail_v3'/row['dataset'];archive.mkdir(parents=True,exist_ok=True)
        frame=output_path(row);before=pd.read_csv(frame)
        target=archive/frame.name
        if target.exists():
            raise RuntimeError('Tail repair archive already exists; inspect instead of overwriting.')
        shutil.move(str(frame),str(target))
        shutil.move(str(frame.with_suffix('.json')),str(target.with_suffix('.json')))
        result=extract_capture(row,config)
        after=pd.read_csv(frame)
        equal=before.equals(after)
        repairs.append(dict(capture_id=row['capture_id'],original_raw_sha256=proof['source_sha256'],
            incomplete_records_excluded=proof['incomplete_records_excluded'],
            normalized_source_sha256=proof['normalized_sha256'],sampled_flow_records_unchanged=equal,
            original_frame_archive=str(target.relative_to(HERE)),new_frame_sha256=digest(frame)))
        extraction=json.loads((HERE/'results/extraction.json').read_text())
        extraction['captures']=[result if r['capture_id']==row['capture_id'] else r for r in extraction['captures']]
        save(HERE/'results/extraction.json',extraction)
        log(f'TAIL REPAIR RE-EXTRACTION {row["capture_id"]}: {proof["incomplete_records_excluded"]} incomplete records excluded; retained flow table unchanged={equal}')
    save(HERE/'results/tail_repair_v3.json',dict(repairs=repairs,raw_sources_unchanged=True))


if __name__=='__main__':
    run()
