import tempfile
from pathlib import Path
from types import SimpleNamespace
import unittest
import pandas as pd
from scapy.all import Ether,IP,UDP,Raw,wrpcap
from features import flow_fingerprint
from packet_label_witness import certify_records


class PacketWitnessTests(unittest.TestCase):
    def fixture(self):
        a=Ether(src='02:00:00:00:00:01',dst='02:00:00:00:00:02')/IP(src='10.0.0.1',dst='10.0.0.2')/UDP(sport=10,dport=20)/Raw(b'abcd')
        b=Ether(src='02:00:00:00:00:02',dst='02:00:00:00:00:01')/IP(src='10.0.0.2',dst='10.0.0.1')/UDP(sport=20,dport=10)/Raw(b'12345678')
        a.time=1.;b.time=3.
        flow=SimpleNamespace(src_ip='10.0.0.1',dst_ip='10.0.0.2',src_port=10,dst_port=20,protocol=17,
            bidirectional_first_seen_ms=1000,bidirectional_last_seen_ms=3000,bidirectional_packets=2,
            bidirectional_bytes=68,src2dst_packets=1,dst2src_packets=1,src2dst_bytes=32,dst2src_bytes=36,
            bidirectional_syn_packets=0,bidirectional_ack_packets=0,bidirectional_fin_packets=0,bidirectional_rst_packets=0)
        key='10.0.0.1:10|10.0.0.2:20|17';capture_id='fixture'
        frame=pd.DataFrame([dict(flow_uid=flow_fingerprint(flow,capture_id),src_ip=flow.src_ip,dst_ip=flow.dst_ip,
            src_port=10,dst_port=20,protocol_number=17,first_ms=1000,last_ms=3000,
            src_packets=1,dst_packets=1,src_bytes=32,dst_bytes=36,min_label=1,key=key)])
        labels=pd.DataFrame(dict(key=[key,key],start_s=[1.,3.],end_s=[1.,3.],label=[1,1]))
        return [a,b],frame,labels,capture_id

    def execute(self,frame_change=None,label_change=None):
        packets,frame,labels,capture=self.fixture()
        if frame_change:
            frame=frame_change(frame)
        if label_change:
            labels=label_change(labels)
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'fixture.pcap';wrpcap(str(path),packets)
            answer,_=certify_records(path,frame,labels,capture,tolerance=.1)
        return answer.iloc[0]

    def test_empty_time_gap_needs_no_invented_traffic(self):
        self.assertTrue(self.execute().certified)

    def test_missing_packet_annotation_rejected(self):
        self.assertFalse(self.execute(label_change=lambda f:f.iloc[:1]).certified)

    def test_original_native_counts_and_fingerprint_required(self):
        self.assertFalse(self.execute(frame_change=lambda f:f.assign(src_bytes=33)).certified)

    def test_packet_binary_conflict_rejected(self):
        self.assertFalse(self.execute(label_change=lambda f:pd.concat([f,f.iloc[:1].assign(label=0)])).certified)


if __name__=='__main__':
    unittest.main(verbosity=2)
