"""Freeze metadata-only capture selection and download complete, checksum-checked files."""
from pathlib import Path
import argparse
import hashlib
import json
import shutil
import time
from concurrent.futures import ThreadPoolExecutor,as_completed
from urllib.parse import quote
import requests
from common import HERE,ROOT,digest,log,save

RAW=ROOT/'data/raw/flow_v2'


def role_assign(rows):
    for dataset in sorted({r['dataset'] for r in rows}):
        pcaps=[r for r in rows if r['dataset']==dataset and r['kind']=='pcap']
        strata=['benign','attack'] if dataset=='cic_iot_2023' else ['mixed']
        for stratum in strata:
            group=[r for r in pcaps if r['stratum']==stratum]
            group.sort(key=lambda r:hashlib.sha256(('v2-fixed-capture-split:'+r['path']).encode()).hexdigest())
            test_n=max(1,round(.2*len(group)))
            cal_n=max(1,round(.2*len(group)))
            for i,row in enumerate(group):
                row['role']='final' if i<test_n else 'calibration' if i<test_n+cal_n else 'train'
    return rows


def make_plan():
    cic=json.loads((HERE/'manifests/hf_somnath0100_CICIoT2023Small.json').read_text())
    selected=[r for r in cic['pcaps'] if '/Benign_Final/' in r['path']]
    singles=['Backdoor_Malware','BrowserHijacking','CommandInjection','DictionaryBruteForce',
        'Recon-HostDiscovery','Recon-OSScan','Recon-PortScan','SqlInjection','XSS']
    for category in singles:
        candidates=[r for r in cic['pcaps'] if r['path'].split('/')[1]==category]
        selected.append(min(candidates,key=lambda r:(r['bytes'],r['path'])))
    for category in ['DDoS-ACK_Fragmentation','Mirai-udpplain']:
        candidates=[r for r in cic['pcaps'] if r['path'].split('/')[1]==category]
        selected.extend(sorted(candidates,key=lambda r:(r['bytes'],r['path']))[:3])
    rows=[]
    for row in selected:
        rows.append(dict(row,dataset='cic_iot_2023',kind='pcap',stratum='benign' if '/Benign_Final/' in row['path'] else 'attack',
            label_provenance='published capture category inherited from original filename; no independent per-packet reverification'))
    ton=json.loads((HERE/'manifests/ton_all_files.json').read_text())
    for row in ton['files']:
        if row['type']!='file' or not row['path'].endswith(('.pcap','.csv')):
            continue
        rows.append(dict(dataset='ton_iot',repository=ton['repository'],revision=ton['revision'],
            path=row['path'],bytes=row['size'],sha256=row.get('lfs',{}).get('oid'),git_oid=row['oid'],
            url='https://huggingface.co/datasets/'+ton['repository']+'/resolve/'+ton['revision']+'/'+quote(row['path'],safe='/'),
            kind='pcap' if row['path'].endswith('.pcap') else 'labels',stratum='mixed',
            label_provenance='must join original-schema Zeek flow labels by tuple/time; never infer TON labels from capture name'))
    rows=role_assign(rows)
    for row in rows:
        row['capture_id']=row['dataset']+':'+row['path']
        row['local_path']=str((RAW/row['dataset']/row['path']).relative_to(ROOT))
    plan=dict(selection='Complete capture files, metadata-only bounded cohort. All four CIC benign captures; nine small attack-category captures; three smallest complete captures in each of two flood categories; all 25 mirrored TON captures and 23 original-schema label tables.',
        sampling_warning='CIC cohort is size-constrained and is not a full-release or natural-prevalence sample. Capture-file disjointness is not proof of acquisition-session independence.',
        mirror_warning='Raw mirror hashes verify downloaded objects, not an independently obtained official PCAP checksum. Raw data will not be redistributed.',
        files=rows,total_bytes=sum(r['bytes'] for r in rows),config_sha256=digest(HERE/'config.json'),
        target_performance_observed=False,comparison_alignment='Historical tables cannot be asserted capture-matched to new flows. A/C and B/D are paired within their tracks; cross-track feature effects remain descriptive unless independently matched.')
    return plan


def download(row):
    output=ROOT/row['local_path'].replace('\\','/')
    output.parent.mkdir(parents=True,exist_ok=True)
    if output.exists():
        actual=digest(output)
        if output.stat().st_size==row['bytes'] and (not row.get('sha256') or actual==row['sha256']):
            log('CACHE VERIFIED '+row['capture_id'])
            return dict(capture_id=row['capture_id'],sha256=actual,bytes=output.stat().st_size)
        raise RuntimeError('Existing finalized source has an unexpected hash: '+str(output))
    partial=output.with_suffix(output.suffix+'.part')
    for attempt in range(4):
        start=partial.stat().st_size if partial.exists() else 0
        try:
            headers={'Accept-Encoding':'identity'}
            if start:
                headers['Range']=f'bytes={start}-'
            with requests.get(row['url'],headers=headers,timeout=(30,120),stream=True) as response:
                response.raise_for_status()
                if start and (response.status_code!=206 or not response.headers.get('Content-Range','').startswith(f'bytes {start}-')):
                    start=0
                last=time.monotonic()
                with partial.open('ab' if start else 'wb') as stream:
                    total=start
                    for chunk in response.iter_content(2**20):
                        if not chunk:
                            continue
                        stream.write(chunk);total+=len(chunk)
                        if time.monotonic()-last>=10:
                            log(f'DOWNLOAD {row["capture_id"]}: {total/2**20:.1f}/{row["bytes"]/2**20:.1f} MiB')
                            last=time.monotonic()
            if partial.stat().st_size!=row['bytes']:
                raise ValueError('Downloaded length does not match pinned metadata.')
            actual=digest(partial)
            if row.get('sha256') and actual!=row['sha256']:
                raise ValueError('Downloaded source SHA256 mismatch.')
            if not row.get('sha256') and row.get('git_oid'):
                data=partial.read_bytes()
                git_hash=hashlib.sha1(('blob '+str(len(data))+'\0').encode()+data).hexdigest()
                if git_hash!=row['git_oid']:
                    raise ValueError('Downloaded Git blob checksum mismatch.')
            partial.replace(output)
            log('DOWNLOAD VERIFIED '+row['capture_id']+' SHA256='+actual)
            return dict(capture_id=row['capture_id'],sha256=actual,bytes=output.stat().st_size)
        except (requests.RequestException,ValueError) as error:
            log(f'DOWNLOAD RETRY {attempt+1}/4 {row["capture_id"]}: {type(error).__name__}')
            if attempt==3:
                raise
            time.sleep(2*(attempt+1))


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan-only',action='store_true');p.add_argument('--kind',choices=['pcap','labels','all'],default='all')
    p.add_argument('--workers',type=int,default=4)
    a=p.parse_args()
    path=HERE/'manifests/capture_plan.json'
    if path.exists():
        plan=json.loads(path.read_text())
        if plan['config_sha256']!=digest(HERE/'config.json'):
            raise RuntimeError('The predeclared configuration changed after capture selection.')
    else:
        plan=make_plan();save(path,plan)
    log(f'FROZEN CAPTURE PLAN: {len(plan["files"])} complete files, {plan["total_bytes"]/2**30:.2f} GiB; target scores not inspected')
    if a.plan_only:
        return
    if shutil.disk_usage(ROOT).free<plan['total_bytes']+20*2**30:
        raise RuntimeError('Insufficient disk space with a 20 GiB reserve.')
    done=[]
    requested=[r for r in sorted(plan['files'],key=lambda r:(r['kind']!='labels',r['bytes'])) if a.kind=='all' or r['kind']==a.kind]
    with ThreadPoolExecutor(max_workers=a.workers) as executor:
        futures=[executor.submit(download,row) for row in requested]
        for future in as_completed(futures):
            done.append(future.result())
            save(HERE/'results'/('downloads_'+a.kind+'.json'),dict(files=done,all_requested_complete=False))
    save(HERE/'results'/('downloads_'+a.kind+'.json'),dict(files=done,all_requested_complete=True))
    log('PASS complete downloads '+a.kind+': '+str(len(done))+' files; no raw data redistributed')


if __name__=='__main__':
    main()
