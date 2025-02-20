import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError


class _BaseTransformer(TransformerMixin, BaseEstimator):

    def __init__(self):
        self.feature_names_in_ = None
        self._feature_names_out = list()
        self._not_fitted = True

    def get_feature_names_out(self) -> list:
        return self._feature_names_out

    def _validate_fitted(self, transformer_name: str):
        if self._not_fitted:
            raise NotFittedError(
                f"This {transformer_name} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this transformer."
            )

    def _validate_keywords(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.feature_names_in_ = X.columns.tolist()
        return None
