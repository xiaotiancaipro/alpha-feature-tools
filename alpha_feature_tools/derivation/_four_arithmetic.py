from typing import List

import pandas as pd
from sklearn.base import TransformerMixin, BaseEstimator
from sklearn.exceptions import NotFittedError

from .._utils import find_unique_subsets


class FourArithmetic(TransformerMixin, BaseEstimator):
    """
    Four arithmetic derivative transformer of continuous variables

    Parameters
    ----------
    feature_names : List[str]
        List of feature names to be transformed

    n : int, default=2
        The number of features calculated each time

    Attributes
    ----------
    feature_names_in_ : List[str]
        List of feature names to be transformed

    Example
    -------
    >>> from alpha_feature_tools.derivation import FourArithmetic

    >>> data = pd.DataFrame({
    ...     "fea_1": [2, 3, 4, 6],
    ...     "fea_2": [1, 1, 3, 4],
    ...     "fea_3": [6, 6, 6, 7]
    ... })

    Derivatied features of two continuous variables uesd four arithmetic operations
    >>> fa = FourArithmetic(["fea_1", "fea_2"])
    >>> fa.fit(data)
    FourArithmetic(feature_names=['fea_1', 'fea_2'])
    >>> fa.transform(data)
       FourArithmetic_fea_1_+_fea_2  ...  FourArithmetic_fea_1_/_fea_2
    0                      2.000000  ...                      2.000000
    1                      3.000000  ...                      3.000000
    2                      1.333333  ...                      1.333333
    3                      1.500000  ...                      1.500000
    [4 rows x 4 columns]

    Derivatied features of three continuous variables uesd four arithmetic operations
    >>> fa = FourArithmetic(["fea_1", "fea_2", "fea_3"], n=3)
    >>> fa.fit(data)
    FourArithmetic(feature_names=['fea_1', 'fea_2', 'fea_3'], n=3)
    >>> fa.transform(data)
       FourArithmetic_fea_1_+_fea_2_+_fea_3  ...  FourArithmetic_fea_1_/_fea_2_/_fea_3
    0                              0.333333  ...                              0.333333
    1                              0.500000  ...                              0.500000
    2                              0.222222  ...                              0.222222
    3                              0.214286  ...                              0.214286
    [4 rows x 4 columns]
    """

    def __init__(
            self,
            feature_names: List[str],
            *,
            n: int = 2
    ):
        self.feature_names = feature_names
        self.n = n
        self.feature_names_in_ = None
        self.__four_arithmetic = ["+", "-", "*", "/"]
        self.__combination_pairs = None
        self.__feature_names_out = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        self._validate_keywords(X, y)

        self.__combination_pairs, self.__feature_names_out = list(), list()
        for subset in find_unique_subsets(self.feature_names_in_, self.n):
            for arithmetic in self.__four_arithmetic:
                self.__combination_pairs.append(subset)
                self.__feature_names_out.append(self.__class__.__name__ + "_" + f"_{arithmetic}_".join(subset))

        return self

    def transform(self, X: pd.DataFrame):

        if self.__feature_names_out is None:
            raise NotFittedError(
                f"This {self.__class__.__name__} instance is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this transformer."
            )

        for combination_pairs, feature_name in zip(self.__combination_pairs, self.__feature_names_out):
            run_list = [f"X[combination_pairs[{_}]]" for _ in range(len(combination_pairs))]
            run_str_list = [f"{_}".join(run_list) for _ in self.__four_arithmetic]
            for run_str in run_str_list:
                X[feature_name] = eval(run_str)

        return X[self.__feature_names_out]

    def get_feature_names_out(self) -> list:
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
