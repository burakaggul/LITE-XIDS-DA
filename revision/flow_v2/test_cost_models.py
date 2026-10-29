import inspect
import tempfile
import unittest
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from cost_models import SourceTransform,class_costs,teacher,student,probability,quantize
from experiment import FAMILIES,source_data,train_source


class CostTests(unittest.TestCase):
    def setUp(self):
        self.y=np.array([0,0,0,0,1,1])
        self.x=pd.DataFrame({'a':[0.,1.,2.,3.,8.,9.],'b':[np.nan,2.,2.,3.,6.,9.]})

    def test_cost_normalization_and_ratio(self):
        one=class_costs(self.y,1);five=class_costs(self.y,5)
        self.assertAlmostEqual(np.mean(five[self.y]),1.)
        self.assertAlmostEqual((five[0]/five[1])/(one[0]/one[1]),5.)

    def test_source_only_imputation(self):
        t=SourceTransform().fit(self.x)
        before=t.imputer.statistics_.copy()
        t.apply(pd.DataFrame({'a':[1e12],'b':[1e12]}))
        np.testing.assert_array_equal(before,t.imputer.statistics_)
        self.assertEqual(before[1],3.)

    def test_identifier_rejected(self):
        with self.assertRaises(ValueError):
            SourceTransform().fit(self.x.assign(src_ip='127.0.0.1'))

    def test_matched_students_and_quantized_serialization(self):
        t=SourceTransform().fit(self.x);x=t.apply(self.x)
        conf=dict(epochs=2,patience=2,batch_size=3,learning_rate=.001,temperature=3.,alpha_hard=.25)
        large=teacher(x,self.y,11,2)
        ordinary=student(x,self.y,x,self.y,11,2,conf)
        kd=student(x,self.y,x,self.y,11,2,conf,large.predict_proba(x))
        self.assertEqual(ordinary.parameter_count,kd.parameter_count)
        self.assertEqual(len(kd.training_history),2)
        quantized=quantize(kd)
        with tempfile.TemporaryDirectory() as tmp:
            file=Path(tmp)/'model.joblib';joblib.dump(quantized,file)
            scores=probability(joblib.load(file),x)
        self.assertTrue(np.isfinite(scores).all())
        self.assertTrue(((scores>=0)&(scores<=1)).all())
        individual=np.array([probability(quantized,row[None,:])[0] for row in x])
        np.testing.assert_allclose(probability(quantized,x),individual,atol=1e-7,rtol=1e-7)

    def test_selection_api_has_no_target(self):
        self.assertNotIn('target',inspect.signature(train_source).parameters)
        body=inspect.getsource(source_data)
        self.assertNotIn("common_partition(name,'final'",body)
        self.assertEqual(len(FAMILIES),4)


if __name__=='__main__':
    unittest.main(verbosity=2)
