from typing import List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder


class BinaryCrossCombination(TransformerMixin, BaseEstimator):
    """
    Cross derivative estimator of two-way combination of categorical variables

    Parameters
    ----------
    feature_names : List[str]
        Features involved in cross derivation

    is_one_hot : bool, default=True
        Does need to perform one-hot encoding on the derived features
    """

    def __init__(self, feature_names: List[str], is_one_hot: bool = True):
        self.__feature_names = feature_names
        self.__is_one_hot = is_one_hot
        self.__encoder = OneHotEncoder(sparse=False)
        self.__combination_pairs = None
        self.__intermediate_feature = None
        self.__feature_names_out = None
        self.feature_names_in_ = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):

        self._check_keywords(X, y)

        self.__combination_pairs, self.__intermediate_feature = list(), list()
        for index, _1 in enumerate(self.__feature_names):
            for _2 in self.__feature_names[index + 1:]:
                self.__combination_pairs.append((_1, _2))
                self.__intermediate_feature.append(_1 + "_&_" + _2)

        self.__feature_names_out = self.__intermediate_feature
        if self.__is_one_hot:
            intermediate_df = self._generate_intermediate_features(X)
            self.__encoder.fit(intermediate_df)
            self.__feature_names_out = self.__encoder.get_feature_names_out()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if self.__feature_names_out is None:
            raise NotFittedError(
                "This BinaryCrossCombination instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this estimator."
            )

        intermediate_df = self._generate_intermediate_features(X)

        if self.__is_one_hot:
            encoded = self.__encoder.transform(intermediate_df)
            return pd.DataFrame(encoded, columns=self.__feature_names_out, index=X.index)

        return intermediate_df[self.__intermediate_feature]

    def get_feature_names_out(self) -> np.ndarray:
        return self.__feature_names_out

    def _check_keywords(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:

        if len(self.__feature_names) < 2:
            raise ValueError("At least two feature columns are required.")

        missing_cols = set(self.__feature_names) - set(X.columns)
        if missing_cols:
            raise ValueError(f"The following columns are missing in the data frame: {missing_cols}")

        self.feature_names_in_ = list(set(self.__feature_names))

        return None

    def _generate_intermediate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        intermediate_data = {
            combined_name: X[_1].astype(str) + "_&_" + X[_2].astype(str)
            for (_1, _2), combined_name in zip(self.__combination_pairs, self.__intermediate_feature)
        }
        return pd.DataFrame(data=intermediate_data, index=X.index)
