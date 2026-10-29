import unittest
import numpy as np
import pandas as pd
from operating_points import source_point,select_cost
from label_flows import canonical_keys,join_labels,observational_identity
from features import FEATURES
import tempfile
from pathlib import Path


class ProtocolTests(unittest.TestCase):
    def test_source_fpr_constraint_including_ties(self):
        y=np.r_[np.zeros(100),np.ones(100)]
        s=np.r_[np.repeat(.7,10),np.repeat(.1,90),np.repeat(.8,80),np.repeat(.2,20)]
        p=source_point(y,s,.01)
        self.assertEqual(p['source_fp'],0)
        self.assertEqual(p['source_recall'],.8)
        self.assertLessEqual(p['source_fpr'],.01)

    def test_constant_score_never_forces_false_alarms(self):
        p=source_point([0,0,1,1],[1,1,1,1],.01)
        self.assertEqual(p['source_fp'],0)
        self.assertEqual(p['source_tp'],0)

    def test_target_metrics_rejected(self):
        p=dict(candidate_id='cost1',benign_cost=1,**source_point([0,1],[0,1],.01))
        with self.assertRaises(ValueError):
            select_cost([dict(p,target_recall=1)])

    def test_selection_has_no_target_label_argument(self):
        a=dict(candidate_id='cost1',benign_cost=1,**source_point([0,1],[0,1],.01))
        b=dict(a,candidate_id='cost20',benign_cost=20)
        self.assertEqual(select_cost([a,b])['benign_cost'],1)

    def test_reverse_flow_key(self):
        a=pd.DataFrame(dict(src_ip=['10.0.0.1'],dst_ip=['10.0.0.2'],src_port=[8],dst_port=[9],protocol_number=[6]))
        b=a.rename(columns={'src_ip':'dst_ip','dst_ip':'src_ip','src_port':'dst_port','dst_port':'src_port'})
        self.assertEqual(canonical_keys(a).iloc[0],canonical_keys(b).iloc[0])

    def test_label_time_containment_and_conflict(self):
        f=pd.DataFrame(dict(flow_uid=['known','conflict','unknown'],src_ip=['a']*3,dst_ip=['b']*3,
            src_port=[1,2,3],dst_port=[4]*3,protocol_number=[6]*3,first_ms=[11000]*3,last_ms=[12000]*3))
        keys=canonical_keys(f)
        labels=pd.DataFrame(dict(key=[keys[0],keys[1],keys[1]],start_s=[10.]*3,end_s=[13.]*3,
            label=[0,0,1],attack=['normal','normal','dos']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.parquet';labels.to_parquet(path,index=False)
            result=join_labels(f,[path],1).set_index('flow_uid')
        self.assertEqual(result.loc['known','joined_label'],0)
        self.assertTrue(pd.isna(result.loc['conflict','joined_label']))
        self.assertTrue(pd.isna(result.loc['unknown','joined_label']))

    def test_segmented_same_label_union(self):
        f=pd.DataFrame(dict(flow_uid=['union','gap','hidden_conflict'],src_ip=['a']*3,dst_ip=['b']*3,
            src_port=[1,2,3],dst_port=[4]*3,protocol_number=[6]*3,first_ms=[10000]*3,last_ms=[13000]*3))
        k=canonical_keys(f)
        labels=pd.DataFrame(dict(key=[k[0],k[0],k[1],k[1],k[2],k[2]],
            start_s=[10.,11.5,10.,12.,10.,11.],end_s=[11.5,13.,11.,13.,13.,12.],
            label=[1,1,1,1,0,1],attack=['dos','dos','dos','dos','normal','dos']))
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'labels.parquet';labels.to_parquet(path,index=False)
            result=join_labels(f,[path],0).set_index('flow_uid')
        self.assertEqual(result.loc['union','joined_label'],1)
        self.assertEqual(result.loc['gap','label_match_status'],'unsupported_time_gap')
        self.assertTrue(pd.isna(result.loc['gap','joined_label']))
        self.assertEqual(result.loc['hidden_conflict','label_match_status'],'conflicting_binary_labels')
        self.assertTrue(pd.isna(result.loc['hidden_conflict','joined_label']))

    def test_native_identity_direction_and_projection_collisions(self):
        f=pd.DataFrame({c:[1.] for c in FEATURES}).assign(src_ip='10.0.0.1',dst_ip='10.0.0.2',
            src_port=8,dst_port=9,first_ms=1000,last_ms=2000,src_packets=1,dst_packets=2,src_bytes=20,dst_bytes=80)
        reverse=f.rename(columns={'src_ip':'dst_ip','dst_ip':'src_ip','src_port':'dst_port','dst_port':'src_port',
            'src_packets':'dst_packets','dst_packets':'src_packets','src_bytes':'dst_bytes','dst_bytes':'src_bytes'})
        self.assertEqual(observational_identity(f).iloc[0],observational_identity(reverse).iloc[0])
        other=f.assign(src_ip='10.0.0.3')
        self.assertNotEqual(observational_identity(f).iloc[0],observational_identity(other).iloc[0])
        self.assertNotEqual(observational_identity(f).iloc[0],observational_identity(f.assign(bytes_max=2.)).iloc[0])


if __name__=='__main__':
    unittest.main(verbosity=2)
