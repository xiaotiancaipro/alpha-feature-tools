import pandas as pd
from sklearn.feature_selection import chi2

from .._base import _BaseTransformer


class ChiSquareSelector(_BaseTransformer):
    """
    A feature selector that computes chi-squared stats and selects features based on
    either the highest chi-squared scores or p-value threshold.

    Parameters
    ----------
    k : int, optional (default=None)
        Number of top features to select based on chi-squared scores.
        If None, p_threshold will be used for selection.

    p_threshold : float, optional (default=None)
        p-value threshold for feature selection.
        If None, k will be used for selection.
    """

    def __init__(self, *, k: float | None = None, p_threshold: float | None = None):
        self.k = k
        self.p_threshold = p_threshold
        self.__feature_stats = dict()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for feature in self.feature_names_in_:
            chi2_value, p_value = chi2(X[feature].values.reshape(-1, 1), y)
            feature_stats = {"feature_name": feature, "chi2_value": chi2_value[0], "p_value": p_value[0]}
            self.__feature_stats[feature] = feature_stats
            if (self.k and (chi2_value[0] > self.k)) or (self.p_threshold and (p_value[0] <= self.p_threshold)):
                self._feature_names_out.append(feature)
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if self.k and self.p_threshold:
            raise ValueError("Only one of k or p_threshold should be specified")
        return X, y

    def _more_tags(self):
        return super()._more_tags().update({"requires_y": True})

    def get_feature_stats(self) -> dict:
        return self.__feature_stats
