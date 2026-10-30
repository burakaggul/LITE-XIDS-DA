"""Certify observed packets, not imaginary traffic during empty flow-time gaps."""
import argparse
from bisect import bisect_right
import hashlib
import ipaddress
import json
from pathlib import Path
import time
import dpkt
import pandas as pd
from scapy.utils import RawPcapReader
from common import HERE,ROOT,digest,log,save


class IntervalLabels:
    def __init__(self,records,tolerance):
        self.intervals={}
        for label in [0,1]:
            ranges=sorted((float(r.start_s)-tolerance,float(r.end_s)+tolerance)
                for r in records.itertuples() if int(r.label)==label)
            merged=[]
            for lo,hi in ranges:
                if merged and lo<=merged[-1][1]:
                    merged[-1]=(merged[-1][0],max(merged[-1][1],hi))
                else:
                    merged.append((lo,hi))
            self.intervals[label]=([a for a,_ in merged],[b for _,b in merged])

    def label_at(self,timestamp):
        matches=[]
        for label,(starts,ends) in self.intervals.items():
            i=bisect_right(starts,timestamp)-1
            if i>=0 and timestamp<=ends[i]:
                matches.append(label)
        return matches[0] if len(matches)==1 else None


def packet_fields(data,link):
    if link==1:
        packet=dpkt.ethernet.Ethernet(data).data
    elif link in [12,101,228,229]:
        packet=dpkt.ip.IP(data) if data[0]>>4==4 else dpkt.ip6.IP6(data)
    elif link==113:
        packet=dpkt.sll.SLL(data).data
    else:
        return None
    if isinstance(packet,dpkt.ip.IP):
        if packet.offset:
            return None
        protocol=packet.p;size=packet.len
    elif isinstance(packet,dpkt.ip6.IP6):
        protocol=getattr(packet,'p',packet.nxt);size=40+packet.plen
    else:
        return None
    transport=packet.data
    if protocol==6 and isinstance(transport,dpkt.tcp.TCP):
        sport,dport,flags=transport.sport,transport.dport,transport.flags
    elif protocol==17 and isinstance(transport,dpkt.udp.UDP):
        sport,dport,flags=transport.sport,transport.dport,0
    elif protocol in [1,58]:
        sport,dport,flags=0,0,0
    else:
        return None
    return (packet.src,packet.dst,int(sport),int(dport),int(protocol)),int(size),int(flags)


def witness_uid(row,state,capture_id):
    values=(str(row.src_ip),str(row.dst_ip),int(row.src_port),int(row.dst_port),int(row.protocol_number),
        int(row.first_ms),int(row.last_ms),state['packets'],state['bytes'],state['src_packets'],state['dst_packets'],
        state['src_bytes'],state['dst_bytes'],state['syn'],state['ack'],state['fin'],state['rst'])
    return hashlib.sha256((capture_id+'|'+repr(values)).encode()).hexdigest()


def certify_records(capture,candidates,intervals,capture_id,tolerance=1.):
    records=list(candidates.drop_duplicates('flow_uid').itertuples());lookup={};states=[]
    labelers={key:IntervalLabels(frame,tolerance) for key,frame in intervals.groupby('key')}
    for i,row in enumerate(records):
        key=(ipaddress.ip_address(row.src_ip).packed,ipaddress.ip_address(row.dst_ip).packed,
            int(row.src_port) if row.protocol_number not in [1,58] else 0,
            int(row.dst_port) if row.protocol_number not in [1,58] else 0,int(row.protocol_number))
        reverse=(key[1],key[0],key[3],key[2],key[4])
        lookup.setdefault(key,[]).append((i,True))
        if reverse!=key:
            lookup.setdefault(reverse,[]).append((i,False))
        states.append(dict(packets=0,bytes=0,src_packets=0,dst_packets=0,src_bytes=0,dst_bytes=0,
            syn=0,ack=0,fin=0,rst=0,first=None,last=None,all_supported=True,ambiguous_attribution=False))
    packets=0;parse_failures=0;last_log=time.monotonic()
    with RawPcapReader(str(capture)) as reader:
        while True:
            try:
                data,meta=reader._read_packet(size=2**32-1)
            except EOFError:
                break
            packets+=1
            if hasattr(meta,'caplen') and len(data)!=meta.caplen:
                raise AssertionError('Witness input contains an incomplete packet record.')
            if hasattr(meta,'tshigh'):
                ticks=(int(meta.tshigh)<<32)|int(meta.tslow);den=int(meta.tsresol);link=int(meta.linktype)
            else:
                den=10**9 if reader.nano else 10**6;ticks=int(meta.sec)*den+int(meta.usec);link=int(reader.linktype)
            ms=ticks*1000//den
            try:
                parsed=packet_fields(data,link)
            except (dpkt.UnpackError,ValueError,IndexError):
                parse_failures+=1;continue
            if parsed is None or parsed[0] not in lookup:
                continue
            key,size,flags=parsed
            matches=[(i,forward) for i,forward in lookup[key] if int(records[i].first_ms)<=ms<=int(records[i].last_ms)]
            if len(matches)>1:
                for i,_ in matches:
                    states[i]['ambiguous_attribution']=True
                continue
            for i,forward in matches:
                row=records[i];state=states[i];direction='src' if forward else 'dst'
                state['packets']+=1;state['bytes']+=size
                state[direction+'_packets']+=1;state[direction+'_bytes']+=size
                state['first']=ms if state['first'] is None else min(state['first'],ms)
                state['last']=ms if state['last'] is None else max(state['last'],ms)
                for name,flag in [('syn',dpkt.tcp.TH_SYN),('ack',dpkt.tcp.TH_ACK),('fin',dpkt.tcp.TH_FIN),('rst',dpkt.tcp.TH_RST)]:
                    state[name]+=int(bool(flags&flag))
                labeler=labelers.get(row.key)
                if labeler is None or labeler.label_at(ticks/den)!=int(row.min_label):
                    state['all_supported']=False
            if time.monotonic()-last_log>20:
                log(f'PACKET WITNESS {capture_id}: {packets:,} packet records inspected')
                last_log=time.monotonic()
    output=[]
    for row,state in zip(records,states):
        identity=witness_uid(row,state,capture_id)
        count_match=all(state[name]==int(getattr(row,name)) for name in ['src_packets','dst_packets','src_bytes','dst_bytes'])
        timing=state['first']==int(row.first_ms) and state['last']==int(row.last_ms)
        valid=identity==row.flow_uid and count_match and timing and state['all_supported'] and not state['ambiguous_attribution']
        output.append(dict(flow_uid=row.flow_uid,capture_id=capture_id,certified=valid,
            witnessed_label=int(row.min_label) if valid else None,all_observed_packets_labeled=state['all_supported'],
            native_counter_match=count_match,native_fingerprint_match=identity==row.flow_uid,
            native_first_last_match=timing,ambiguous_attribution=state['ambiguous_attribution'],observed_packets=state['packets']))
    return pd.DataFrame(output),dict(total_packet_records=packets,unparsed_packet_records=parse_failures)


def prepare():
    import duckdb
    from label_flows import canonical_keys
    from label_schema_v3 import normalize_keys
    data=json.loads((HERE/'manifests/common_dataset.json').read_text())
    if data.get('label_schema_version')!='v3':
        raise RuntimeError('The normalized v3 annotation schema is required.')
    unknown=pd.read_parquet(HERE/'prepared/unresolved_labels.parquet')
    candidates=unknown[unknown.label_match_status.eq('unsupported_time_gap')].copy()
    candidates['key']=normalize_keys(canonical_keys(candidates))
    schema=json.loads((HERE/'manifests/label_schema_v3.json').read_text())
    files=[str(HERE/r['output']) for r in schema['files']]
    connection=duckdb.connect();connection.execute('SET threads=4');connection.register('keys',candidates[['key']].drop_duplicates())
    intervals=connection.execute('SELECT l.* FROM read_parquet(?) l SEMI JOIN keys k ON l.key=k.key',[files]).df()
    directory=HERE/'cache/packet_witness_v3';directory.mkdir(parents=True,exist_ok=True)
    captures=[]
    for capture,frame in candidates.groupby('capture_id'):
        name=hashlib.sha256(capture.encode()).hexdigest()[:20]
        frame.to_csv(directory/(name+'_flows.csv.gz'),index=False)
        intervals[intervals.key.isin(frame.key)].to_csv(directory/(name+'_labels.csv.gz'),index=False)
        captures.append(dict(capture_id=capture,key=name,candidates=len(frame),
            flows_sha256=digest(directory/(name+'_flows.csv.gz')),
            labels_sha256=digest(directory/(name+'_labels.csv.gz'))))
    save(directory/'manifest.json',dict(captures=captures,source_data_manifest_sha256=digest(HERE/'manifests/common_dataset.json'),
        schema_manifest_sha256=digest(HERE/'manifests/label_schema_v3.json'),tolerance_seconds=1.,
        rule='Every actual packet must have unanimous original interval/point support; reconstructed native counts, first/last times and original flow fingerprint including TCP flags must match exactly. Empty time gaps do not acquire invented packets or labels.',
        class_inference=False,code_sha256=digest(__file__)))
    log(f'PACKET WITNESS INPUTS READY: {len(candidates):,} time-gap candidates in {len(captures)} captures; no labels changed')


def run_capture(capture_id):
    directory=HERE/'cache/packet_witness_v3';manifest=json.loads((directory/'manifest.json').read_text())
    if manifest['code_sha256']!=digest(__file__):
        raise RuntimeError('Packet-witness code changed after input preparation.')
    row=next(r for r in manifest['captures'] if r['capture_id']==capture_id)
    if digest(directory/(row['key']+'_flows.csv.gz'))!=row['flows_sha256'] or digest(directory/(row['key']+'_labels.csv.gz'))!=row['labels_sha256']:
        raise AssertionError('A packet-witness input changed after its freeze.')
    metadata=HERE/'cache/chronological/ton_iot'/(row['key']+'.json')
    source=json.loads(metadata.read_text());capture=Path(source['normalized_path'])
    if digest(capture)!=source['normalized_sha256']:
        raise AssertionError('Normalized raw capture changed.')
    frame=pd.read_csv(directory/(row['key']+'_flows.csv.gz'))
    intervals=pd.read_csv(directory/(row['key']+'_labels.csv.gz'))
    log(f'PACKET WITNESS START {capture_id}: {len(frame):,} candidate flows; exact packet/flow certificate')
    answer,stats=certify_records(capture,frame,intervals,capture_id,manifest['tolerance_seconds'])
    out=HERE/'results/packet_witness_v3';out.mkdir(exist_ok=True)
    answer.to_csv(out/(row['key']+'.csv'),index=False)
    save(out/(row['key']+'.json'),dict(capture_id=capture_id,candidates=len(answer),certified=int(answer.certified.sum()),
        normalized_source_sha256=source['normalized_sha256'],input_manifest_sha256=digest(directory/'manifest.json'),
        code_sha256=digest(__file__),**stats))
    log(f'PACKET WITNESS FINISHED {capture_id}: {int(answer.certified.sum()):,}/{len(answer):,} certified; rejected records remain unknown')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--capture');parser.add_argument('--all',action='store_true')
    arguments=parser.parse_args()
    if arguments.prepare:
        prepare()
    elif arguments.all:
        manifest=json.loads((HERE/'cache/packet_witness_v3/manifest.json').read_text())
        for row in manifest['captures']:
            run_capture(row['capture_id'])
        log('ALL PACKET WITNESSES COMPLETE: verified certificates only; rejected flows remain unknown')
    else:
        run_capture(arguments.capture)
