from typing import List

import pandas as pd

from .._base import _BaseTransformer


class GroupStatistics(_BaseTransformer):
    """
    Grouped Statistical Feature Derivative Transformer

    Parameters
    ----------
    group_features : List[List[str]]
        List of lists of features to group and calculate statistics on
        e.g. [["feature1", "feature2"], ["feature3", "feature4"]]
        will group feature1 and feature2 together and calculate mean and median
        and feature3 and feature4 together and calculate mean and median
        and add these new features to the dataset
    """

    def __init__(self, *, group_features: List[List[str]]):
        self.group_features = group_features
        self.__unique_group_features = set()
        self.__statistics = ["mean", "median"]
        self.temp = list()
        self.group_stats = dict()
        super().__init__()

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if not isinstance(self.group_features, list):
            raise ValueError("group_features must be a list")
        for feature_group in self.group_features:
            for feature in feature_group:
                self.__unique_group_features.add(feature)
                if feature not in self.feature_names_in_:
                    raise ValueError(f"Group feature {feature_group} not in input features")
        return X, y

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        feature_names_in_ = list(set(self.feature_names_in_) - self.__unique_group_features)
        for feature_grouped in self.group_features:
            for feature in feature_names_in_:
                grouped = X.groupby(feature_grouped)[feature].agg(self.__statistics)
                for stat, val in grouped.to_dict().items():
                    new_feature = f"{self.__class__.__name__}_{feature}_by_{'&'.join(feature_grouped)}_grouped_{stat}"
                    self.temp.append(feature_grouped)
                    self._feature_names_out.append(new_feature)
                    self.group_stats[new_feature] = {
                        "&".join(list(map(str, k))) if isinstance(k, tuple) else k: v
                        for k, v in val.items()
                    }
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature_grouped, feature in zip(self.temp, self._feature_names_out):
            temp_feature = "&".join(feature_grouped)
            if temp_feature not in X.columns:
                rum_str = " +\"&\"+ ".join([f"X['{_}'].astype(str)" for _ in feature_grouped])
                X[temp_feature] = eval(rum_str)
            X[feature] = X[temp_feature].map(lambda x: self.group_stats[feature][x])
        return X[self._feature_names_out]
