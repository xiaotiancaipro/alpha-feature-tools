import pandas as pd
from abc import ABC, abstractmethod
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


class _BaseTransformer(TransformerMixin, BaseEstimator, ABC):

    def __init__(self):
        self.feature_names_in_ = None
        self._feature_names_out = list()
        self._not_fitted = True

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self._validate_keywords(X, y)
        self._fit(X, y)
        self._not_fitted = False
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self._not_fitted:
            raise NotFittedError(
                "This instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this transformer."
            )
        return self._transform(X)

    def get_feature_names_out(self) -> list:
        return self._feature_names_out

    @abstractmethod
    def _fit(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        return None

    @abstractmethod
    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame()

    def _validate_keywords(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"X must be a pandas DataFrame, got {type(X)}")
        self.feature_names_in_ = X.columns.tolist()
        return None
