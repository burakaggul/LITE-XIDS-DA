"""Normalize capture order with Wireshark, preserving exact packets and recorded timestamps."""
import hashlib
import json
import math
import os
import re
from pathlib import Path
import shutil
import subprocess
from scapy.utils import RawPcapReader
from common import HERE,ROOT,digest,log,save

VERSION='capture-chronology-v1'
MODULUS=1<<256


def fingerprint(path):
    count=0;captured=0;wire=0;total=0;xor=0;behind=0;backsteps=0;latest=None;previous=None
    max_lookback=0;resolution=None;incomplete=[];last_complete_end=None;tail_bytes=0;tail_sha=None
    with RawPcapReader(str(path)) as reader:
        file_format=type(reader).__name__
        last_complete_end=reader.f.tell()
        while True:
            record_start=reader.f.tell()
            try:
                packet,meta=reader._read_packet(size=2**32-1)
            except EOFError:
                break
            if hasattr(meta,'caplen') and len(packet)!=int(meta.caplen):
                incomplete.append(dict(offset=record_start,expected_captured_bytes=int(meta.caplen),
                    available_captured_bytes=len(packet),missing_bytes=int(meta.caplen)-len(packet),
                    available_payload_sha256=hashlib.sha256(packet).hexdigest()))
                continue
            if hasattr(meta,'tshigh'):
                ticks=(int(meta.tshigh)<<32)|int(meta.tslow);den=int(meta.tsresol)
                link=int(meta.linktype)
            else:
                den=10**9 if reader.nano else 10**6
                ticks=int(meta.sec)*den+int(meta.usec);link=int(reader.linktype)
            common=math.gcd(ticks,den);stamp=(ticks//common,den//common)
            # Cross multiplication preserves nanosecond precision without float conversion.
            if latest is not None and ticks*latest[1]<latest[0]*den:
                behind+=1;max_lookback=max(max_lookback,(latest[0]/latest[1])-(ticks/den))
            if previous is not None and ticks*previous[1]<previous[0]*den:
                backsteps+=1
            if latest is None or ticks*latest[1]>=latest[0]*den:
                latest=(ticks,den)
            previous=(ticks,den)
            h=hashlib.sha256(repr((stamp,int(meta.wirelen),link,len(packet))).encode()+b'\0'+packet).digest()
            integer=int.from_bytes(h,'big');total=(total+integer)%MODULUS;xor^=integer
            count+=1;captured+=len(packet);wire+=int(meta.wirelen)
            resolution=den if resolution in [None,den] else 'mixed'
            last_complete_end=reader.f.tell()
        if file_format=='RawPcapReader':
            tail_bytes=Path(path).stat().st_size-last_complete_end
            if tail_bytes:
                reader.f.seek(last_complete_end);tail_sha=hashlib.sha256(reader.f.read()).hexdigest()
                if not incomplete:
                    if tail_bytes>=16:
                        raise RuntimeError('Unparsed classic-PCAP tail is not merely a short record header.')
                    incomplete.append(dict(offset=last_complete_end,truncated_record_header_bytes=tail_bytes))
    return dict(packets=count,captured_bytes=captured,wire_bytes=wire,
        packet_timestamp_multiset_sum256=f'{total:064x}',packet_timestamp_multiset_xor256=f'{xor:064x}',
        adjacent_backsteps=backsteps,packets_behind_running_max=behind,max_lookback_seconds=max_lookback,
        timestamp_resolution=resolution,file_format=file_format,
        incomplete_records=len(incomplete),incomplete_record_details=incomplete,
        quarantined_raw_tail_bytes=tail_bytes,quarantined_raw_tail_sha256=tail_sha,
        digest_scope='All complete packet records, full captured payload, exact recorded timestamp, wire length and link type. Incomplete raw tail remains preserved in the pinned source.')


def executable():
    found=shutil.which('reordercap')
    env=os.environ.copy()
    if not found:
        local=Path.home()/'.local/opt/litexids-wireshark/root'
        candidate=local/'usr/bin/reordercap'
        if not candidate.exists():
            raise RuntimeError('Install/extract the official Wireshark reordercap utility; do not rewrite packet timestamps.')
        found=str(candidate)
        libs=str(local/'usr/lib/x86_64-linux-gnu')
        env['LD_LIBRARY_PATH']=libs+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    return found,env


def normalize(source,output):
    source=Path(source);output=Path(output);meta=output.with_suffix('.json')
    sha=digest(source)
    if meta.exists():
        old=json.loads(meta.read_text())
        normalized=Path(old['normalized_path'])
        if old['version']!=VERSION or old['source_sha256']!=sha or not normalized.exists() or digest(normalized)!=old['normalized_sha256']:
            raise RuntimeError('Chronology cache/source integrity mismatch.')
        return old
    output.parent.mkdir(parents=True,exist_ok=True)
    tool,env=executable();temporary=output.with_suffix('.reordering')
    if temporary.exists():
        raise RuntimeError('Interrupted normalization artifact exists; preserve and inspect before restarting: '+str(temporary))
    log('CHRONOLOGY CHECK '+source.name)
    result=subprocess.run([tool,'-n',str(source),str(temporary)],env=env,capture_output=True,text=True,check=True)
    counts=re.search(r'(\d+) frames, (\d+) out of order',result.stdout)
    if counts is None:
        log('REORDERCAP DIAGNOSTIC '+repr((result.stdout,result.stderr)))
        raise RuntimeError('Unrecognized reordercap packet-count diagnostic; preserve output for inspection.')
    reported_frames,out_of_order=map(int,counts.groups())
    truncated='cut short in the middle of a packet' in result.stderr.lower()
    sorted_copy=out_of_order>0 or truncated
    if not sorted_copy and temporary.exists():
        # Wireshark 4.2 may leave an empty header-only output even with -n.
        temporary.unlink()
    evidence=None
    if sorted_copy:
        if truncated and out_of_order==0:
            if temporary.exists():
                temporary.unlink()
            result=subprocess.run([tool,str(source),str(temporary)],env=env,capture_output=True,text=True,check=True)
        before=fingerprint(source);after=fingerprint(temporary)
        invariants=['packets','captured_bytes','wire_bytes','packet_timestamp_multiset_sum256','packet_timestamp_multiset_xor256']
        if any(before[k]!=after[k] for k in invariants) or after['packets_behind_running_max']!=0:
            raise AssertionError('Sorted capture does not preserve packet/timestamp multiset or chronology.')
        if truncated and not before['incomplete_records']:
            raise RuntimeError('Capture parser reports truncation not explained by the verified classic-PCAP tail audit.')
        if after['incomplete_records']:
            raise AssertionError('Normalized capture still contains an incomplete record.')
        temporary.replace(output);normalized=output
        evidence=dict(original=before,normalized=after)
    else:
        normalized=source
    version=subprocess.run([tool,'--version'],env=env,capture_output=True,text=True,check=True).stdout.splitlines()[0]
    record=dict(version=VERSION,source_sha256=sha,normalized_path=str(normalized),normalized_sha256=digest(normalized),
        sorted_copy=sorted_copy,tool=version,tool_stdout=result.stdout.strip(),tool_stderr=result.stderr.strip(),
        preservation=evidence,timestamps_modified=False,packet_bytes_modified=False,
        reported_frames=reported_frames,reported_out_of_order=out_of_order,
        structural_validation='complete-records-v1.1',incomplete_records_excluded=evidence['original']['incomplete_records'] if evidence else 0,
        preservation_scope='All complete packets and their recorded timestamps are preserved. An incomplete raw tail, if present, is excluded and reported without modifying the pinned original.',
        limitation='Order follows original capture timestamps, not an independent verification that acquisition clocks were synchronized.')
    save(meta,record)
    log('CHRONOLOGY PASS '+source.name+(': normalized; complete packet/timestamp multiset preserved' if sorted_copy else ': already ordered; original bytes used')+
        '; incomplete records quarantined='+str(record['incomplete_records_excluded']))
    return record


def ensure(row):
    source=ROOT/row['local_path'].replace('\\','/')
    key=hashlib.sha256(row['capture_id'].encode()).hexdigest()[:20]
    output=HERE/'cache/chronological'/row['dataset']/(key+'.pcap')
    return normalize(source,output)
