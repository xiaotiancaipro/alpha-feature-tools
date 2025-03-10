import numpy as np
import pandas as pd

from .._base import _BaseTransformer


class NLP(_BaseTransformer):

    def __init__(self, is_tf_idf: bool = True):
        self.is_tf_idf = is_tf_idf
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for feature in self.feature_names_in_:
            self._feature_names_out.append(f"{self.__class__.__name__}_{feature}")
        if self.is_tf_idf:
            self._feature_names_out = [f"{_}_tf_idf" for _ in self._feature_names_out]
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        document_sum = 0
        word_dict = dict()
        for feature, name in zip(self.feature_names_in_, self._feature_names_out):
            X[name] = X[feature].map(lambda x: 1 if x != 0 else 0)
            word_dict[feature] = X[name].sum()
            document_sum += word_dict[feature]
        if self.is_tf_idf:
            for feature, name in zip(self.feature_names_in_, self._feature_names_out):
                X[name] = X[name] * ((np.log(document_sum / word_dict[feature])) + 1)
        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        return X, y
