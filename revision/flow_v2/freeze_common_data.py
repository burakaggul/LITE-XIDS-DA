"""Bind the common data snapshot to its annotation evidence before model fitting."""
import argparse
import json
from datetime import datetime,timezone
from common import HERE,digest,log,save


def snapshot():
    manifest=HERE/'manifests/common_dataset.json'
    data=json.loads(manifest.read_text())
    if not data['quality_gate_passed']:
        raise RuntimeError('Cannot freeze common training data while the label quality gate fails.')
    files=[manifest,HERE/'manifests/capture_plan.json',HERE/'results/prepared_audit.json']
    if data.get('official_event_sources_sha256'):
        for name,key in [('official_event_sources.json','official_event_sources_sha256'),
            ('official_annotation_amendment.json','annotation_amendment_sha256')]:
            path=HERE/'manifests'/name
            if digest(path)!=data[key]:
                raise AssertionError('Official annotation provenance changed.')
            files.append(path)
    files.extend(sorted((HERE/'prepared').glob('*/*.parquet')))
    if len(list((HERE/'prepared').glob('*/*.parquet')))!=6:
        raise AssertionError('Exactly six fixed data partitions are required.')
    return {str(p.relative_to(HERE)).replace('\\','/'):digest(p) for p in files}


def run(stage):
    path=HERE/'manifests/common_data_freeze.json';current=snapshot()
    if path.exists():
        stored=json.loads(path.read_text())
        if stored['sha256']!=current:
            raise AssertionError('Common data/annotation snapshot changed after freeze.')
        log('COMMON DATA FREEZE VERIFIED: all partition and provenance hashes unchanged')
    elif stage=='verify':
        raise RuntimeError('No common data freeze exists.')
    else:
        if any((HERE/'models/common').rglob('frozen.json')) or (HERE/'results/evaluation_start_common.json').exists():
            raise RuntimeError('A new common data freeze must precede fitting/evaluation.')
        save(path,dict(sha256=current,frozen_utc=datetime.now(timezone.utc).isoformat(),
            common_performance_seen=False,hashing_final_data_is_not_model_selection=True))
        log('COMMON DATA FROZEN: partitions, label evidence and physical audit pinned before fitting')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--stage',choices=['freeze','verify'],required=True)
    run(parser.parse_args().stage)
