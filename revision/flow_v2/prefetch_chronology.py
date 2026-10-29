"""Byte-identical Linux-local staging avoids slow per-packet I/O through the WSL mount."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from common import HERE,ROOT,digest,log,save
from chronology import normalize


def run():
    scratch=Path.home()/'.cache/litexids-flow-v2-chronology'
    (scratch/'raw').mkdir(parents=True,exist_ok=True);(scratch/'ordered').mkdir(exist_ok=True)
    plan=json.loads((HERE/'manifests/capture_plan.json').read_text())
    rows=sorted([r for r in plan['files'] if r['kind']=='pcap'],key=lambda r:-r['bytes'])
    done=[]
    for row in rows:
        key=hashlib.sha256(row['capture_id'].encode()).hexdigest()[:20]
        directory=HERE/'cache/chronological'/row['dataset'];directory.mkdir(parents=True,exist_ok=True)
        expected=directory/(key+'.json')
        if expected.exists() or (directory/(key+'.reordering')).exists():
            continue
        raw=ROOT/row['local_path'].replace('\\','/')
        if not raw.exists():
            raise RuntimeError('Only complete pinned captures may be staged.')
        source_sha=digest(raw)
        if row.get('sha256') and row['sha256']!=source_sha:
            raise AssertionError('Original pinned source checksum mismatch.')
        local=scratch/'raw'/(key+'.pcap')
        if not local.exists():
            temporary=local.with_suffix('.copying');shutil.copyfile(raw,temporary)
            if digest(temporary)!=source_sha:
                raise AssertionError('Linux-local byte-identical staging failed.')
            temporary.replace(local)
        elif digest(local)!=source_sha:
            raise AssertionError('Existing staged copy checksum mismatch.')
        log('LOCAL STAGING PASS '+row['capture_id']+': exact source bytes retained; no method/role/label change')
        record=normalize(local,scratch/'ordered'/(key+'.pcap'))
        record['staging']='Byte-identical Linux-local copy; original pinned Windows source remains unchanged.'
        record['capture_id']=row['capture_id']
        if record['source_sha256']!=source_sha:
            raise AssertionError('Chronology proof does not match pinned source.')
        staging=directory/(key+'.prefetch.json');save(staging,record)
        try:
            # Do not overwrite a concurrent extractor's own completed normalization.
            os.link(staging,expected)
            log('CHRONOLOGY PREFETCH PUBLISHED '+row['capture_id'])
        except FileExistsError:
            log('CHRONOLOGY PREFETCH NOT REPLACED '+row['capture_id']+': extractor already published its proof')
        finally:
            staging.unlink()
        done.append(row['capture_id'])
        save(HERE/'results/chronology_prefetch.json',dict(captures=done,complete=False,raw_sources_modified=False))
    save(HERE/'results/chronology_prefetch.json',dict(captures=done,complete=True,raw_sources_modified=False))
    log('LOCAL CHRONOLOGY PREFETCH COMPLETE: same algorithm and packet/timestamp preservation, faster local I/O only')


if __name__=='__main__':
    run()
