import tempfile
import unittest
from pathlib import Path
import pandas as pd
from label_flows import canonical_keys,join_labels
from official_events import event_intervals


class OfficialEventTests(unittest.TestCase):
    def events(self):
        return pd.DataFrame(dict(ts=[10,11],src_ip=['a','a'],dst_ip=['b','b'],
            src_port=[1,2],dst_port=[4,4],proto=['tcp','udp'],type=['scanning','normal']))

    def test_explicit_types_and_zero_duration(self):
        result=event_intervals(self.events())
        self.assertEqual(result.label.tolist(),[1,0])
        self.assertTrue(result.start_s.eq(result.end_s).all())
        with self.assertRaises(ValueError):
            event_intervals(self.events().assign(type='unverified'))

    def test_point_does_not_invent_long_flow_support(self):
        flows=pd.DataFrame(dict(flow_uid=['short','long'],src_ip=['a']*2,dst_ip=['b']*2,
            src_port=[1]*2,dst_port=[4]*2,protocol_number=[6]*2,
            first_ms=[10500]*2,last_ms=[10800,20000]))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'events.parquet';event_intervals(self.events()).to_parquet(path,index=False)
            result=join_labels(flows,[path],1).set_index('flow_uid')
        self.assertEqual(result.loc['short','joined_label'],1)
        self.assertTrue(pd.isna(result.loc['long','joined_label']))

    def test_conflicting_original_normal_still_rejected(self):
        flow=pd.DataFrame(dict(flow_uid=['conflict'],src_ip=['a'],dst_ip=['b'],src_port=[1],dst_port=[4],
            protocol_number=[6],first_ms=[10000],last_ms=[10000]))
        normal=pd.DataFrame(dict(key=canonical_keys(flow),start_s=[9.],end_s=[11.],label=[0],attack=['normal']))
        with tempfile.TemporaryDirectory() as directory:
            a=Path(directory)/'normal.parquet';b=Path(directory)/'events.parquet'
            normal.to_parquet(a,index=False);event_intervals(self.events()).to_parquet(b,index=False)
            result=join_labels(flow,[a,b],1).iloc[0]
        self.assertEqual(result.label_match_status,'conflicting_binary_labels')
        self.assertTrue(pd.isna(result.joined_label))


if __name__=='__main__':
    unittest.main(verbosity=2)
