import pandas as pd

from .._base import _BaseTransformer


class OutlierProcessor(_BaseTransformer):
    """
    Outlier Detection and Processing Transformer

    Parameters
    ----------
    method : str
        Method for outlier detection, 'std' (3-sigma rule) or 'box' (IQR method)

    treatment : str
        Method for outlier treatment, 'mean' (mean correction), 'cap' (capping), or 'none' (no treatment)
    """

    def __init__(self, method="std", treatment="cap"):
        self.method = method
        self.treatment = treatment
        self.__stats = dict()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for feature in self.feature_names_in_:
            feature_stats = dict()
            feature_stats["mean"] = X[feature].mean()
            if self.method == "std":
                feature_stats["std"] = X[feature].std()
                feature_stats["lower"] = feature_stats["mean"] - 3 * feature_stats["std"]
                feature_stats["upper"] = feature_stats["mean"] + 3 * feature_stats["std"]
            if self.method == "box":
                q1 = X[feature].quantile(0.25)
                q3 = X[feature].quantile(0.75)
                iqr = q3 - q1
                feature_stats["lower"] = q1 - 1.5 * iqr
                feature_stats["upper"] = q3 + 1.5 * iqr
            self.__stats[feature] = feature_stats
        self._feature_names_out = self.feature_names_in_
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X_transformed = X.copy()
        for feature in self.feature_names_in_:
            stats = self.__stats[feature]
            lower, upper = stats["lower"], stats["upper"]
            if self.treatment == "mean":
                mask = (X_transformed[feature] < lower) | (X_transformed[feature] > upper)
                X_transformed.loc[mask, feature] = stats["mean"]
            if self.treatment == "cap":
                X_transformed[feature] = X_transformed[feature].clip(lower=lower, upper=upper)
        return X_transformed[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if self.method not in ["std", "box"]:
            raise ValueError("method 必须是 'std' 或 'box'")
        if self.treatment not in ["mean", "cap", "none"]:
            raise ValueError("treatment 必须是 'mean', 'cap' 或 'none'")
        return X, y
