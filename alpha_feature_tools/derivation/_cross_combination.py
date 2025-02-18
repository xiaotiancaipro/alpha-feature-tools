import itertools
from typing import List

import numpy as np
import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError
from sklearn.preprocessing import OneHotEncoder

from ._base import StrBase


class CrossCombination(TransformerMixin, BaseEstimator):
    """
    Cross derivative transformer of categorical variables

    Parameters
    ----------
    feature_names : List[str]
        Features involved in cross derivation
    
    n : int, default=2
        Number of features involved in cross derivation, default is binary cross combination

    is_one_hot : bool, default=False
        Does need to perform one-hot encoding on the derived features

    Examples
    --------
    >>> from alpha_feature_tools.derivation import CrossCombination

    >>> data = pd.DataFrame({
    ...     "fea_1": ["Q", "W", "W", "Q"],
    ...     "fea_2": ["A", "B", "A", "C"],
    ...     "fea_3": [1, 1, 3, 4],
    ...     "fea_4": [6, 6, 6, 7]
    ... })

    Binary cross combination

    >>> cc = CrossCombination(feature_names=["fea_1", "fea_2", "fea_3"])
    >>> cc.fit(data)
    CrossCombination(feature_names=['fea_1', 'fea_2', 'fea_3'])
    >>> cc.transform(data)
      fea_1_&_fea_2 fea_2_&_fea_3 fea_1_&_fea_3
    0         Q_&_A         A_&_1         Q_&_1
    1         W_&_B         B_&_1         W_&_1
    2         W_&_A         A_&_3         W_&_3
    3         Q_&_C         C_&_4         Q_&_4

    Binary cross combination and execute ont-hot encoding

    >>> cc = CrossCombination(feature_names=["fea_1", "fea_2", "fea_3"], is_one_hot=True)
    >>> cc.fit(data)
    CrossCombination(feature_names=['fea_1', 'fea_2', 'fea_3'], is_one_hot=True)
    >>> cc.transform(data)
       fea_1_&_fea_2_Q_&_A  ...  fea_1_&_fea_3_W_&_3
    0                  1.0  ...                  0.0
    1                  0.0  ...                  0.0
    2                  0.0  ...                  1.0
    3                  0.0  ...                  0.0
    [4 rows x 12 columns]

    Three variable cross combination

    >>> cc = CrossCombination(feature_names=["fea_1", "fea_2", "fea_3", "fea_4"], n=3)
    >>> cc.fit(data)
    CrossCombination(feature_names=['fea_1', 'fea_2', 'fea_3', 'fea_4'], n=3)
    >>> cc.transform(data)
      fea_1_&_fea_2_&_fea_3  ... fea_1_&_fea_2_&_fea_4
    0             Q_&_A_&_1  ...             Q_&_A_&_6
    1             W_&_B_&_1  ...             W_&_B_&_6
    2             W_&_A_&_3  ...             W_&_A_&_6
    3             Q_&_C_&_4  ...             Q_&_C_&_7
    [4 rows x 4 columns]
    """

    def __init__(
            self,
            feature_names: List[str],
            *,
            n: int = 2,
            is_one_hot: bool = False
    ):
        self.feature_names = feature_names
        self.n = n
        self.is_one_hot = is_one_hot
        self.feature_names_in_ = None
        self.__encoder = OneHotEncoder(sparse=False)
        self.__combination_pairs = None
        self.__intermediate_feature = None
        self.__feature_names_out = None

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):

        self._validate_keywords(X, y)

        self.__combination_pairs, self.__intermediate_feature = list(), list()
        for subset in self.__find_unique_subsets(self.feature_names, self.n):
            self.__combination_pairs.append(subset)
            self.__intermediate_feature.append(StrBase.SEP.join(subset))

        self.__feature_names_out = self.__intermediate_feature
        if self.is_one_hot:
            intermediate_df = self._generate_intermediate_features(X)
            self.__encoder.fit(intermediate_df)
            self.__feature_names_out = self.__encoder.get_feature_names_out()

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if self.__feature_names_out is None:
            raise NotFittedError(
                "This CrossCombination instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this transformer."
            )

        intermediate_df = self._generate_intermediate_features(X)

        if self.is_one_hot:
            encoded = self.__encoder.transform(intermediate_df)
            return pd.DataFrame(encoded, columns=self.__feature_names_out, index=X.index)

        return intermediate_df[self.__intermediate_feature]

    def get_feature_names_out(self) -> np.ndarray:
        return self.__feature_names_out

    def _validate_keywords(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:

        feature_names_set = set(self.feature_names)

        if len(feature_names_set) < self.n:
            raise ValueError(f"At least {self.n} feature columns are required.")

        missing_cols = feature_names_set - set(X.columns)
        if missing_cols:
            raise ValueError(f"The following columns are missing in the data frame: {missing_cols}.")

        self.feature_names_in_ = list(feature_names_set)

        return None

    def _generate_intermediate_features(self, X: pd.DataFrame) -> pd.DataFrame:
        intermediate_data = dict()
        for combination_pairs, combined_name in zip(self.__combination_pairs, self.__intermediate_feature):
            run_list = [f"X[combination_pairs[{_}]].astype(str)" for _ in range(len(combination_pairs))]
            run_str = f" + \"{StrBase.SEP}\" + ".join(run_list)
            intermediate_data[combined_name] = eval(run_str)
        return pd.DataFrame(data=intermediate_data, index=X.index)

    def __find_unique_subsets(self, data: list, n: int = 2) -> list:
        combinations = itertools.combinations(data, n)
        unique_combinations = {tuple(sorted(_)) for _ in combinations}
        return list(unique_combinations)
