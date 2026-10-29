"""Source-fitted full native control; never fit categories on held-out data."""
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline


class NativeTransform:
    def fit(self,source):
        self.numeric=list(source.select_dtypes(include='number').columns)
        other=[c for c in source.columns if c not in self.numeric]
        self.categorical=[c for c in other if source[c].nunique(dropna=True)<=30]
        self.dropped=[c for c in other if c not in self.categorical]
        self.columns=self.numeric+self.categorical
        self.transformer=ColumnTransformer([
            ('numeric',make_pipeline(SimpleImputer(strategy='median',keep_empty_features=True),
                StandardScaler()),self.numeric),
            ('categorical',OneHotEncoder(handle_unknown='ignore',sparse_output=False),self.categorical)],
            remainder='drop',sparse_threshold=0.)
        self.transformer.fit(self._frame(source))
        return self

    def _frame(self,frame):
        work=frame[self.columns].copy()
        if self.categorical:
            work[self.categorical]=work[self.categorical].fillna('missing').astype(str)
        return work

    def apply(self,frame,source=False):
        return np.asarray(self.transformer.transform(self._frame(frame)),dtype=float)
