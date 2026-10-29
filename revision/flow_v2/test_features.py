import unittest
from types import SimpleNamespace
import tempfile
from pathlib import Path
import sys
from features import FEATURES,common_features,HashReservoir


class FeatureTests(unittest.TestCase):
    def test_physical_units(self):
        flow=SimpleNamespace(bidirectional_duration_ms=2000,bidirectional_packets=3,
            bidirectional_bytes=180,bidirectional_min_ps=40,bidirectional_max_ps=80,
            protocol=6,bidirectional_mean_piat_ms=1000)
        row=common_features(flow)
        self.assertEqual(row['duration_s'],2)
        self.assertEqual(row['bytes_mean'],60)
        self.assertEqual(row['byte_rate_s'],90)
        self.assertEqual(row['iat_mean_s'],1)
        self.assertEqual(set(row),set(FEATURES))

    def test_missing_measurements_not_fabricated(self):
        import math
        f=SimpleNamespace(bidirectional_duration_ms=0,bidirectional_packets=1,bidirectional_bytes=40,
            bidirectional_min_ps=40,bidirectional_max_ps=40,protocol=17,bidirectional_mean_piat_ms=0)
        r=common_features(f)
        self.assertTrue(math.isnan(r['byte_rate_s']))
        self.assertTrue(math.isnan(r['iat_mean_s']))

    def test_sampling_order_independent(self):
        a,b=HashReservoir(7),HashReservoir(7)
        for value in range(100):
            a.offer(str(value),lambda value=value:dict(v=value))
        for value in reversed(range(100)):
            b.offer(str(value),lambda value=value:dict(v=value))
        self.assertEqual(a.records(),b.records())

    @unittest.skipUnless(sys.platform=='linux','Actual NFStream runtime is the pinned Linux environment')
    def test_actual_pcap_has_identical_ip_byte_definition(self):
        from scapy.all import Ether,IP,UDP,Raw,wrpcap
        from nfstream import NFStreamer
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'unit.pcap'
            a=Ether(src='00:00:00:00:00:01',dst='00:00:00:00:00:02')/IP(src='10.0.0.1',dst='10.0.0.2')/UDP(sport=10,dport=20)/Raw(b'abcd')
            b=Ether(src='00:00:00:00:00:02',dst='00:00:00:00:00:01')/IP(src='10.0.0.2',dst='10.0.0.1')/UDP(sport=20,dport=10)/Raw(b'12345678')
            a.time=1600000000.;b.time=1600000002.
            wrpcap(str(path),[a,b])
            flows=list(NFStreamer(source=str(path),accounting_mode=1,statistical_analysis=True,n_dissections=0,n_meters=1))
            self.assertEqual(len(flows),1)
            row=common_features(flows[0])
            self.assertEqual(row['bytes_total'],68)
            self.assertEqual(row['bytes_min'],32)
            self.assertEqual(row['bytes_max'],36)
            self.assertEqual(row['iat_mean_s'],2)


if __name__=='__main__':
    unittest.main(verbosity=2)
