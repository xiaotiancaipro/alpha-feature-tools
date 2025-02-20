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

    >>> fa = FourArithmetic()
    >>> fa.fit(data[["fea_1", "fea_2"]])
    FourArithmetic()
    >>> fa.transform(data[["fea_1", "fea_2"]])
       FourArithmetic_fea_1_+_fea_2  ...  FourArithmetic_fea_1_/_fea_2
    0                      2.000000  ...                      2.000000
    1                      3.000000  ...                      3.000000
    2                      1.333333  ...                      1.333333
    3                      1.500000  ...                      1.500000
    [4 rows x 4 columns]

    Derivatied features of three continuous variables uesd four arithmetic operations

    >>> fa = FourArithmetic(n=3)
    >>> fa.fit(data[["fea_1", "fea_2", "fea_3"]])
    FourArithmetic(n=3)
    >>> fa.transform(data[["fea_1", "fea_2", "fea_3"]])
       FourArithmetic_fea_1_+_fea_2_+_fea_3  ...  FourArithmetic_fea_1_/_fea_2_/_fea_3
    0                              0.333333  ...                              0.333333
    1                              0.500000  ...                              0.500000
    2                              0.222222  ...                              0.222222
    3                              0.214286  ...                              0.214286
    [4 rows x 4 columns]
    """

    def __init__(
            self,
            *,
            n: int = 2
    ):
        self.n = n
        self.feature_names_in_ = None
        self.__four_arithmetic = ["+", "-", "*", "/"]
        self.__combination_pairs = None
        self.__feature_names_out = None

    def fit(self, X: pd.DataFrame, y: pd.Series = None):

        self._validate_keywords(X, y)

        self.__combination_pairs, self.__feature_names_out = list(), list()
        a = find_unique_subsets(self.feature_names_in_, self.n)
        for subset in a:
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

        index = 0
        for combination_pairs, feature_name in zip(self.__combination_pairs, self.__feature_names_out):
            run_list = [f"X[\"{_}\"]" for _ in combination_pairs]
            run_str = f" {self.__four_arithmetic[index % len(self.__four_arithmetic)]} ".join(run_list)
            X[feature_name] = eval(run_str)
            index += 1

        return X[self.__feature_names_out]

    def get_feature_names_out(self) -> list:
        return self.__feature_names_out

    def _validate_keywords(self, X: pd.DataFrame, y: pd.Series | None = None) -> None:
        self.feature_names_in_ = X.columns.tolist()
        if len(self.feature_names_in_) < self.n:
            raise ValueError(f"At least {self.n} feature columns are required.")
        return None
