"""Import author-hosted event annotations; no duration or benign labels are inferred."""
import argparse
from collections import Counter
from datetime import datetime,timezone
import json
from pathlib import Path
import shutil
from urllib.parse import quote
import zipfile
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from common import HERE,ROOT,digest,log,save
from label_flows import canonical_keys

ATTACKS={'scanning','ddos','dos','ransomware','backdoor','injection','xss','password','mitm'}
DIRECTORY='/personal/z5025758_ad_unsw_edu_au/Documents/TON_IoT datasets/SecuityEvents_GroundTruth_datasets/SecurityEvents_Network_datasets/'


def event_intervals(frame):
    types=frame['type'].astype(str).str.strip().str.lower()
    unexpected=set(types)-ATTACKS-{'normal'}
    if unexpected:
        raise ValueError('Undocumented official event type: '+str(sorted(unexpected)))
    timestamp=pd.to_numeric(frame.ts,errors='raise')
    if not np.isfinite(timestamp).all():
        raise ValueError('An official event timestamp is missing/nonfinite.')
    result=pd.DataFrame(dict(key=canonical_keys(frame,True),start_s=timestamp.astype(float),
        end_s=timestamp.astype(float),label=types.ne('normal').astype('int64'),attack=types))
    if result.key.str.endswith('|-1').any():
        raise ValueError('Undocumented official protocol code.')
    return result


def import_archive(archive):
    if any((HERE/'models/common').rglob('frozen.json')):
        raise RuntimeError('Annotation amendments are forbidden after common source fitting.')
    destination=ROOT/'data/raw/flow_v2/ton_official_events'
    destination.mkdir(parents=True,exist_ok=True)
    index=HERE/'official_event_index';index.mkdir(exist_ok=True)
    expected={f'GroundTruth_Network_{i}.csv' for i in range(1,19)}
    original=HERE/'manifests/common_dataset.json'
    amendment=HERE/'manifests/official_annotation_amendment.json'
    if not amendment.exists():
        save(amendment,dict(version='official-event-points-v1',declared_utc=datetime.now(timezone.utc).isoformat(),
            previous_data_manifest_sha256=digest(original),previous_label_coverage=.9363480372665488,
            reason='Acquire original author-hosted annotations to diagnose unsupported Zeek-table labels before any common-flow model fitting.',
            source='https://research.unsw.edu.au/projects/toniot-datasets',
            event_rule='Exact orientation-invariant tuple; start=end=official event timestamp; no event duration imputed. Apply the already frozen 1s tolerance and unanimous overlapping binary/full-union-time support jointly with original Zeek intervals.',
            normal_rule='Only explicit type=normal is benign; absence from an attack table is not benign evidence.',
            required_coverage=.95,clock_tolerance_seconds=1.,capture_roles_unchanged=True,
            training_code_and_hyperparameters_unchanged=True,common_model_performance_inspected=False,
            new_source_count=18,code_sha256=digest(__file__)))
    records=[]
    with zipfile.ZipFile(archive) as source:
        members=[r for r in source.infolist() if not r.is_dir()]
        if {Path(r.filename).name for r in members}!=expected or len(members)!=18:
            raise ValueError('Official archive does not contain exactly the 18 observed CSV files.')
        if source.testzip() is not None:
            raise ValueError('Official archive CRC failure.')
        for member in members:
            name=Path(member.filename).name;raw=destination/name;out=index/(Path(name).stem+'.parquet')
            if not raw.exists():
                temporary=raw.with_suffix('.csv.part')
                with source.open(member) as reader,temporary.open('wb') as writer:
                    shutil.copyfileobj(reader,writer)
                temporary.replace(raw)
            if raw.stat().st_size!=member.file_size:
                raise ValueError('Official CSV size changed.')
            writer=None;count=0;types=Counter()
            try:
                for frame in pd.read_csv(raw,chunksize=200000,encoding='utf-8-sig',low_memory=False):
                    table=event_intervals(frame);types.update(table.attack)
                    arrow=pa.Table.from_pandas(table,preserve_index=False)
                    if writer is None:
                        writer=pq.ParquetWriter(str(out)+'.tmp',arrow.schema,compression='zstd')
                    writer.write_table(arrow);count+=len(table)
            finally:
                if writer is not None:
                    writer.close()
            Path(str(out)+'.tmp').replace(out)
            record=dict(file=name,raw_path=str(raw.relative_to(ROOT)).replace('\\','/'),
                raw_sha256=digest(raw),bytes=member.file_size,zip_crc32=member.CRC,
                index_path=str(out.relative_to(HERE)).replace('\\','/'),index_sha256=digest(out),rows=count,
                explicit_types=dict(types),official_url='https://unsw-my.sharepoint.com'+quote(DIRECTORY+name,safe='/'))
            records.append(record)
            log(f'OFFICIAL EVENT INDEX PASS {name}: {count:,} timestamped explicit annotations, no duration fabricated')
    save(HERE/'manifests/official_event_sources.json',dict(files=records,complete=len(records)==18,
        official_source=True,archive_sha256=digest(archive),archive_crc_verified=True,
        authority='Downloaded through the public author-hosted UNSW SharePoint folder linked by the official TON_IoT project.',
        upstream_version='Observed file modification date 8 February 2024; SHA256 pins this downloaded snapshot, not an asserted historical release identity.',
        annotation_amendment_sha256=digest(amendment),code_sha256=digest(__file__)))
    log('OFFICIAL EVENT IMPORT COMPLETE: 18 original files separately pinned; primary capture cohort and ML protocol unchanged')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    import_archive(parser.parse_args().archive)
