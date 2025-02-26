import pandas as pd
from sklearn.feature_selection import f_classif

from .._base import _BaseTransformer


class ANOVAFSelector(_BaseTransformer):
    """
    ANOVA feature selector, feature screening based on F value

    Parameters
    ----------
    f: float, optional
        F value threshold, default None

    p_threshold: float, optional
        p value threshold, default None
    """

    def __init__(self, *, f: float | None = None, p_threshold: float | None = None):
        self.f = f
        self.p_threshold = p_threshold
        self.__feature_stats = dict()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for feature in self.feature_names_in_:
            f_value, p_value = f_classif(X[feature].values.reshape(-1, 1), y)
            feature_stats = {"feature_name": feature, "f_value": f_value[0], "p_value": p_value[0]}
            self.__feature_stats[feature] = feature_stats
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature, stats in self.__feature_stats.items():
            if (
                    ((self.f is not None) and (stats["f_value"] > self.f)) or
                    ((self.p_threshold is not None) and (stats["p_value"] <= self.p_threshold))
            ):
                self._feature_names_out.append(feature)
        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if self.f and self.p_threshold:
            raise ValueError("Only one of f or p_threshold should be specified")
        return X, y

    def _more_tags(self):
        return super()._more_tags().update({"requires_y": True})

    def set_f(self, f: float | None) -> None:
        self.f = f
        return None

    def set_p_threshold(self, p_threshold: float | None) -> None:
        self.p_threshold = p_threshold
        return None

    def get_feature_stats(self) -> dict:
        return self.__feature_stats
