import unittest
import numpy as np
import pandas as pd
from protocol import (alarm_rule, conservative_alarm_predict, group_split, hashes,
    select_on_source, audit_projection, binary_labels)


class ProtocolTests(unittest.TestCase):
    def test_constant_scores_do_not_force_all_alarms(self):
        scores = np.ones(20000)
        rule = alarm_rule(scores, .01)
        self.assertAlmostEqual(rule.tie_probability, .01)
        self.assertLess(abs(rule.predict(scores, np.arange(len(scores)), 42).mean()-.01), .004)
        self.assertEqual(conservative_alarm_predict(scores, rule).sum(), 0)

    def test_budget_endpoints(self):
        scores = np.arange(20.)
        self.assertEqual(alarm_rule(scores, 0).predict(scores, range(20), 42).sum(), 0)
        self.assertEqual(alarm_rule(scores, 1).predict(scores, range(20), 42).sum(), 20)

    def test_boundary_expected_budget(self):
        scores = np.array([0., 0., 0., 1., 1., 1., 2., 2.])
        for b in [.01, .1, .2, .5, .8]:
            r = alarm_rule(scores, b)
            expected = ((scores > r.threshold).sum() +
                (scores == r.threshold).sum()*r.tie_probability)/len(scores)
            self.assertAlmostEqual(expected, b)

    def test_selection_rejects_test_metrics(self):
        with self.assertRaises(ValueError):
            select_on_source([dict(id='a', source_f1=.5, source_ba=.5, external_f1=.9)])
        self.assertEqual(select_on_source([dict(id='b', source_f1=.5, source_ba=.5),
            dict(id='a', source_f1=.5, source_ba=.5)]), 'a')

    def test_projection_collisions_retained_and_grouped(self):
        native = pd.DataFrame({'a': np.arange(100), 'b': np.arange(100)})
        mapped = pd.DataFrame({'a': np.arange(100)//4})
        y = np.arange(100)%2
        audit = audit_projection(native, mapped, y)
        self.assertEqual(audit['rows'], 100)
        self.assertEqual(audit['projection_collision_keys'], 25)
        a, b = group_split(mapped, y, .2, 42)
        self.assertFalse(set(hashes(mapped.iloc[a])) & set(hashes(mapped.iloc[b])))

    def test_invalid_labels_fail(self):
        with self.assertRaises(ValueError):
            binary_labels(['attack', 'nan'])
        with self.assertRaises(ValueError):
            binary_labels(['benign', 'unlabeled'])
        self.assertEqual(binary_labels([0.0,1.0]).tolist(), [0,1])

    def test_identity_fields_distinguish_original_records(self):
        from data_revision import prepare
        data=pd.DataFrame({'src_ip':['192.0.2.1','192.0.2.2'], 'dst_ip':['192.0.2.3']*2,
            'src_port':[100,101], 'dst_port':[80,80], 'proto':['tcp']*2, 'duration':[1.,1.],
            'src_bytes':[100.,100.], 'dst_bytes':[100.,100.], 'src_pkts':[2,2], 'dst_pkts':[2,2],
            'label':[0,0], 'type':['normal']*2, '_uid':['a','b'], '_capture':['c']*2, '_attack':['normal']*2})
        result=prepare('ton_iot',data,2,11)
        self.assertEqual(result['audit']['rows'],2)
        self.assertEqual(result['audit']['projection_collision_keys'],1)
        data.loc[1,'proto']='udp'
        encoded=prepare('ton_iot',data,2,11)
        self.assertFalse(np.array_equal(encoded['native'].iloc[0],encoded['native'].iloc[1]))


if __name__ == '__main__':
    unittest.main()
