import unittest
import numpy as np
import pandas as pd
from run_revision import Transform
from protocol import alarm_rule,stable_uniform
from native_revision import NativeTransform


class TransformTests(unittest.TestCase):
    def test_native_categories_fit_only_on_source_training(self):
        training=pd.DataFrame({'n':[1.,2.,np.nan],'service':['dns','http','dns']})
        evaluation=pd.DataFrame({'n':[1e9],'service':['new-unseen-service']})
        t=NativeTransform().fit(training)
        categories=t.transformer.named_transformers_['categorical'].categories_[0].copy()
        t.apply(evaluation)
        np.testing.assert_array_equal(categories,t.transformer.named_transformers_['categorical'].categories_[0])
        self.assertNotIn('new-unseen-service',categories)

    def test_evaluation_does_not_fit_preprocessing(self):
        train=pd.DataFrame({'a':[1.,2.,np.nan,4.],'b':[0.,1.,2.,3.]})
        adaptation=pd.DataFrame({'a':[10.,20.,30.],'b':[1.,1.,1.]})
        evaluation=pd.DataFrame({'a':[1e12,np.nan],'b':[-1e12,0.]})
        for variant in ['baseline','robust','robust_coral','robust_gate_iw']:
            t=Transform(variant).fit(train,adaptation)
            mean=t.scaler.mean_.copy()
            median=t.imputer.statistics_.copy()
            t.apply(evaluation)
            np.testing.assert_array_equal(t.scaler.mean_,mean)
            np.testing.assert_array_equal(t.imputer.statistics_,median)

    def test_hash_rule_reproducible_and_label_free(self):
        ids=['file:1','file:2','file:3']
        np.testing.assert_array_equal(stable_uniform(ids,11),stable_uniform(ids,11))
        r=alarm_rule([0.,1.,1.],.5)
        np.testing.assert_array_equal(r.predict([1.,1.,1.],ids,11),r.predict([1.,1.,1.],ids,11))


if __name__=='__main__':
    unittest.main()
