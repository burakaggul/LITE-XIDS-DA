"""A single physical feature definition for both raw-capture sources."""
import hashlib
import math
import heapq

FEATURES=['duration_s','protocol_number','packet_total','bytes_total','bytes_mean',
    'bytes_min','bytes_max','byte_rate_s','tcp','udp','icmp','iat_mean_s']


def common_features(flow):
    duration=flow.bidirectional_duration_ms/1000.
    packets=int(flow.bidirectional_packets)
    total=int(flow.bidirectional_bytes)
    return dict(duration_s=duration,protocol_number=int(flow.protocol),packet_total=packets,
        bytes_total=total,bytes_mean=total/packets if packets else math.nan,
        bytes_min=float(flow.bidirectional_min_ps),bytes_max=float(flow.bidirectional_max_ps),
        byte_rate_s=total/duration if duration>0 else math.nan,
        tcp=int(flow.protocol==6),udp=int(flow.protocol==17),icmp=int(flow.protocol in [1,58]),
        iat_mean_s=flow.bidirectional_mean_piat_ms/1000. if packets>1 else math.nan)


def flow_fingerprint(flow,capture_id):
    fields=['src_ip','dst_ip','src_port','dst_port','protocol','bidirectional_first_seen_ms',
        'bidirectional_last_seen_ms','bidirectional_packets','bidirectional_bytes',
        'src2dst_packets','dst2src_packets','src2dst_bytes','dst2src_bytes',
        'bidirectional_syn_packets','bidirectional_ack_packets','bidirectional_fin_packets',
        'bidirectional_rst_packets']
    value=capture_id+'|'+repr(tuple(getattr(flow,key,None) for key in fields))
    return hashlib.sha256(value.encode()).hexdigest()


class HashReservoir:
    """Order-independent hash-priority sampling, preserving repeated retained records."""
    def __init__(self,capacity):
        if capacity<1:
            raise ValueError('Positive capacity required.')
        self.capacity=capacity
        self.heap=[]
        self.seen=0

    def offer(self,uid,make_record):
        priority=int.from_bytes(hashlib.blake2b(uid.encode(),digest_size=16).digest(),'big')
        self.seen+=1
        if len(self.heap)<self.capacity:
            heapq.heappush(self.heap,(-priority,uid,self.seen,make_record()))
        elif priority < -self.heap[0][0]:
            heapq.heapreplace(self.heap,(-priority,uid,self.seen,make_record()))

    def records(self):
        return [item[3] for item in sorted(self.heap,key=lambda r:(r[1],r[2]))]
