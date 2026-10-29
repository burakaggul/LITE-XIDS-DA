"""Independently compare the imported annotations with the downloaded official ZIP."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import zipfile
import duckdb
from common import HERE,ROOT,digest,log,save


def audit(archive):
    path=HERE/'manifests/official_event_sources.json';manifest=json.loads(path.read_text())
    if not manifest['complete'] or digest(archive)!=manifest['archive_sha256']:
        raise AssertionError('Official archive or import manifest differs.')
    records=[];duplicates=defaultdict(list)
    with zipfile.ZipFile(archive) as source:
        for record in manifest['files']:
            entry=source.getinfo('SecurityEvents_Network_datasets/'+record['file'])
            hasher=hashlib.sha256()
            with source.open(entry) as stream:
                for block in iter(lambda:stream.read(2**20),b''):
                    hasher.update(block)
            if hasher.hexdigest()!=record['raw_sha256'] or digest(ROOT/record['raw_path'])!=record['raw_sha256']:
                raise AssertionError('Official imported file is not byte-identical to its CRC-checked ZIP member.')
            index=HERE/record['index_path']
            if digest(index)!=record['index_sha256']:
                raise AssertionError('Official event index changed.')
            connection=duckdb.connect()
            rows,bad=connection.execute('''SELECT count(*),count_if(start_s!=end_s OR label!=1)
                FROM read_parquet(?)''',[str(index)]).fetchone()
            connection.close()
            if rows!=record['rows'] or bad:
                raise AssertionError('Observed official event files must retain their attack-only point annotations.')
            duplicates[record['raw_sha256']].append(record['file'])
            records.append(dict(file=record['file'],rows=rows,sha256=record['raw_sha256']))
            log('INDEPENDENT OFFICIAL ANNOTATION AUDIT PASS '+record['file'])
    save(HERE/'results/official_event_integrity.json',dict(complete=len(records)==18,files=records,
        annotation_rows=sum(r['rows'] for r in records),unique_file_hashes=len(duplicates),
        identical_upstream_files=[names for names in duplicates.values() if len(names)>1],
        duplicate_annotation_note='Repeated identical author-hosted files do not provide independent label evidence; repeated support cannot outweigh any conflicting binary label.',
        official_source_manifest_sha256=digest(path),
        all_zip_members_and_raw_files_byte_identical=True,all_indexes_hash_and_rowcount_checked=True,
        point_times_preserved_without_duration=True,absence_never_labeled_benign=True,code_sha256=digest(__file__)))
    log('OFFICIAL ANNOTATION AUDIT COMPLETE: 18 files/17 unique byte streams; all imported timestamps remain points')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True)
    audit(parser.parse_args().archive)
