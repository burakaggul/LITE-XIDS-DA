import itertools
import unittest
import numpy as np
from missing_label_bounds import metric_bounds,confusion_metrics


class MissingLabelBoundsTests(unittest.TestCase):
    def test_bounds_are_sharp_by_exhaustive_assignments(self):
        known=(3,2,1,4)
        for positives in range(4):
            for negatives in range(4):
                prediction=[1]*positives+[0]*negatives
                all_metrics=[]
                for labels in itertools.product([0,1],repeat=len(prediction)):
                    tn,fp,fn,tp=known
                    for truth,alarm in zip(labels,prediction):
                        tn+=int(truth==0 and alarm==0);fp+=int(truth==0 and alarm==1)
                        fn+=int(truth==1 and alarm==0);tp+=int(truth==1 and alarm==1)
                    all_metrics.append(confusion_metrics(tn,fp,fn,tp))
                bounds=metric_bounds(*known,positives,negatives)
                for metric in all_metrics[0]:
                    values=[m[metric] for m in all_metrics]
                    self.assertAlmostEqual(bounds[metric+'_lower'],min(values))
                    self.assertAlmostEqual(bounds[metric+'_upper'],max(values))

    def test_zero_missing_labels_collapses_to_observed_metric(self):
        observed=confusion_metrics(10,2,3,8)
        bounds=metric_bounds(10,2,3,8,0,0)
        for name,value in observed.items():
            self.assertAlmostEqual(bounds[name+'_lower'],value)
            self.assertAlmostEqual(bounds[name+'_upper'],value)

    def test_invalid_counts_or_missing_known_class_rejected(self):
        with self.assertRaises(ValueError):
            metric_bounds(1,-1,1,1,0,0)
        with self.assertRaises(ValueError):
            metric_bounds(0,0,1,1,0,0)


if __name__=='__main__':
    unittest.main(verbosity=2)
