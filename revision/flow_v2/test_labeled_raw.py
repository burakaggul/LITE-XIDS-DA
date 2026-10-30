import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
import labeled_raw_experiment as experiment
from features import FEATURES


class LabeledRawTests(unittest.TestCase):
    def test_scope_explicitly_retains_failed_full_traffic_gate(self):
        self.assertIn('Preserved separately',experiment.SPEC['original_full_traffic_gate'])
        self.assertIn('Unknown labels are never imputed',experiment.SPEC['missingness'])

    def test_source_loader_has_no_target_input_and_keeps_ids_out(self):
        import inspect
        self.assertEqual(list(inspect.signature(experiment.source_data).parameters),['dataset','seed'])
        with tempfile.TemporaryDirectory() as root:
            path=Path(root)/'data'/'source';path.mkdir(parents=True)
            for role in ['train','calibration']:
                frame=pd.DataFrame({name:[1.,2.,3.,4.] for name in FEATURES})
                frame=frame.assign(label=[0,1,0,1],flow_uid=[role+str(i) for i in range(4)],
                    raw_identity=[role+str(i) for i in range(4)],capture_id=role)
                frame.to_parquet(path/(role+'.parquet'),index=False)
            with patch.object(experiment,'RUN',Path(root)):
                train,cal,audit=experiment.source_data('source',11)
                self.assertEqual(audit['train_rows'],4)
                self.assertFalse(set(train.raw_identity)&set(cal.raw_identity))
                self.assertEqual(list(train[FEATURES]),FEATURES)

    def test_failed_data_gate_is_not_set_true_by_conditional_study(self):
        self.assertIn('not full-traffic performance',experiment.SPEC['estimand'])
        self.assertIn('including nonrandom missingness',experiment.SPEC['missingness'])


if __name__=='__main__':
    unittest.main(verbosity=2)
