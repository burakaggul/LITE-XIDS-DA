import sys
import tempfile
import unittest
import dpkt
from pathlib import Path
from scapy.all import Ether,IP,UDP,Raw,wrpcap
from chronology import fingerprint,normalize


@unittest.skipUnless(sys.platform.startswith('linux'),'Wireshark user-local utility is installed in Linux runtime')
class ChronologyTests(unittest.TestCase):
    def test_sort_preserves_packet_bytes_timestamps_and_duplicates(self):
        packets=[]
        for time,payload in [(3.,b'a'),(1.,b'b'),(1.,b'b'),(2.,b'c')]:
            p=Ether(src='02:00:00:00:00:01',dst='02:00:00:00:00:02')/IP(src='10.0.0.1',dst='10.0.0.2')/UDP(sport=1,dport=2)/Raw(payload)
            p.time=time;packets.append(p)
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'unordered.pcap';output=Path(directory)/'ordered.pcap'
            wrpcap(str(source),packets);r=normalize(source,output)
            self.assertTrue(r['sorted_copy']);self.assertEqual(r['preservation']['normalized']['packets'],4)
            self.assertEqual(fingerprint(output)['packets_behind_running_max'],0)
            self.assertFalse(r['timestamps_modified']);self.assertFalse(r['packet_bytes_modified'])
            self.assertEqual(normalize(source,output)['source_sha256'],r['source_sha256'])

    def test_ordered_capture_keeps_original(self):
        p=Ether(src='02:00:00:00:00:01',dst='02:00:00:00:00:02')/IP()/UDP();p.time=1.
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'ordered.pcap';output=Path(directory)/'copy.pcap'
            wrpcap(str(source),[p]);r=normalize(source,output)
            self.assertFalse(r['sorted_copy']);self.assertEqual(Path(r['normalized_path']),source)
            self.assertFalse(output.exists())

    def test_incomplete_tail_quarantined_without_fabrication(self):
        p=Ether(src='02:00:00:00:00:01',dst='02:00:00:00:00:02')/IP()/UDP()/Raw(b'complete')
        p.time=1.;q=p.copy();q.time=2.
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'cut.pcap';output=Path(directory)/'ordered.pcap'
            wrpcap(str(source),[p,q])
            with source.open('r+b') as stream:
                stream.truncate(source.stat().st_size-2)
            original=source.read_bytes();r=normalize(source,output)
            self.assertEqual(source.read_bytes(),original)
            self.assertEqual(r['incomplete_records_excluded'],1)
            self.assertEqual(r['preservation']['original']['incomplete_record_details'][0]['missing_bytes'],2)
            self.assertEqual(r['preservation']['normalized']['packets'],1)
            self.assertEqual(r['preservation']['normalized']['incomplete_records'],0)

    def test_legitimate_capture_snaplen_is_not_file_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'short_snapshot.pcap'
            with source.open('wb') as stream:
                dpkt.pcap.Writer(stream,linktype=1)
                header=dpkt.pcap.LEPktHdr if sys.byteorder=='little' else dpkt.pcap.PktHdr
                stream.write(bytes(header(tv_sec=1,tv_usec=0,caplen=3,len=100))+b'abc')
            record=fingerprint(source)
            self.assertEqual(record['packets'],1)
            self.assertEqual(record['captured_bytes'],3)
            self.assertEqual(record['wire_bytes'],100)
            self.assertEqual(record['incomplete_records'],0)


if __name__=='__main__':
    unittest.main(verbosity=2)
