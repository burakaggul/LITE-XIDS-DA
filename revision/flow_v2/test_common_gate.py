import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import experiment
import freeze_common_data


class CommonGateTests(unittest.TestCase):
    def directory(self,root):
        path=Path(root);(path/'manifests').mkdir()
        (path/'manifests/common_dataset.json').write_text(json.dumps({'quality_gate_passed':False}),encoding='utf-8')
        return path

    def test_source_fit_api_stops_before_loading_partitions(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(experiment,'HERE',self.directory(root)),patch.object(experiment,'common_partition') as partition:
                with self.assertRaisesRegex(RuntimeError,'quality gate'):
                    experiment.source_data('common','ton_iot',42)
                partition.assert_not_called()

    def test_data_cannot_be_frozen_after_failed_label_gate(self):
        with tempfile.TemporaryDirectory() as root:
            with patch.object(freeze_common_data,'HERE',self.directory(root)):
                with self.assertRaisesRegex(RuntimeError,'quality gate'):
                    freeze_common_data.snapshot()


if __name__=='__main__':
    unittest.main(verbosity=2)
