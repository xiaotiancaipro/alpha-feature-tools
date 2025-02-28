from abc import ABC, abstractmethod

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


class _BaseTransformer(TransformerMixin, BaseEstimator, ABC):

    def __init__(self):
        self.feature_names_in_ = None
        self._feature_names_out = list()
        self._not_fitted = True

    def fit(self, X, y=None):
        X_, y_ = self._validate_keywords(X, y)
        self._fit(X_, y_)
        self._not_fitted = False
        return self

    def transform(self, X):
        if self._not_fitted:
            raise NotFittedError(
                "This instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this transformer."
            )
        if not isinstance(X, pd.DataFrame):
            X_transform = pd.DataFrame(X)
            if len(X_transform.columns.tolist()) != len(self.feature_names_in_):
                raise ValueError("Columns in X do not match columns in fit.")
            X_transform.columns = self.feature_names_in_
        else:
            if X.columns.tolist() != self.feature_names_in_:
                raise ValueError("Columns in X do not match columns in fit.")
            X_transform = X.copy()
        return self._transform(X_transform).values

    def get_feature_names_out(self) -> list:
        return self._feature_names_out

    def _validate_keywords(self, X, y=None) -> tuple:
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        self.feature_names_in_ = X.columns.tolist()
        return X, y

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y=None) -> None:
        return None

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()
