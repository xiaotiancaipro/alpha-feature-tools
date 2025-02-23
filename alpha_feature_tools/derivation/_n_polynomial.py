import pandas as pd

from .._base import _BaseTransformer
from .._utils import find_unique_subsets


class NPolynomial(_BaseTransformer):
    """
    NPolynomial derivative transformer of continuous variables

    Generates polynomial features of order n_ from the given features.
    The generated features are of the form f1**n1 * f2**n2 * ... * fn**nn, where n1 + n2 + ... + nn = n_.

    Parameters
    ----------
    n : int, default=1
        The number of features to select for polynomial combination.

    n_ : int, default=3
        The order of the polynomial features to generate.

    Examples
    --------
    >>> from alpha_feature_tools.derivation import NPolynomial

    >>> X = pd.DataFrame(
    ...     data=[
    ...         [1, 2, 3, 4, 5],
    ...         [2, 3, 4, 5, 6]
    ...     ],
    ...     columns=["a", "b", "c", "d", "e"]
    ... )
    >>> n_poly = NPolynomial(n=2, n_=2)
    >>> n_poly.fit(X)
    >>> n_poly
    NPolynomial(n=2, n_=2)
    >>> n_poly.transform(X)
       NPolynomial_a**1*b**1  ...  NPolynomial_d**1*e**1
    0                      2  ...                     20
    1                      6  ...                     30
    [2 rows x 15 columns]
    """

    def __init__(self, *, n: int = 1, n_: int = 3):
        self.n = n
        self.n_ = n_
        self.__combination_pairs = list()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for subset in find_unique_subsets(self.feature_names_in_, self.n):
            feature_poly_all = [{key: val} for val in range(1, self.n_ + 1) for key in subset]
            for feature_poly in feature_poly_all:
                if (sum(feature_poly.values()) == self.n_) and (feature_poly not in self.__combination_pairs):
                    self.__combination_pairs.append(feature_poly)
                    feature_name_out = [f"{key}**{val}" for key, val in feature_poly.items()][0]
                    self._feature_names_out.append(f"{self.__class__.__name__}_{feature_name_out}")
                for feature_poly_ in feature_poly_all:
                    if feature_poly.keys() == feature_poly_.keys():
                        continue
                    if sum((sum(feature_poly.values()), sum(feature_poly_.values()))) != self.n_:
                        continue
                    combination_pairs = {**feature_poly, **feature_poly_}
                    if combination_pairs in self.__combination_pairs:
                        continue
                    self.__combination_pairs.append(combination_pairs)
                    feature_name_out = "*".join([f"{key}**{val}" for key, val in combination_pairs.items()])
                    self._feature_names_out.append(f"{self.__class__.__name__}_{feature_name_out}")
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for combination_pairs, feature_name in zip(self.__combination_pairs, self._feature_names_out):
            temp_feature_name = list()
            for key, val in combination_pairs.items():
                if feature_name in temp_feature_name:
                    X[feature_name] *= X[key] ** val
                    continue
                X[feature_name] = X[key] ** val
                temp_feature_name.append(feature_name)
        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if len(self.feature_names_in_) < self.n:
            raise ValueError(f"At least {self.n} feature columns are required.")
        if self.n_ < 2:
            raise ValueError(f"n_ must be greater than 1.")
        if self.n > self.n_:
            raise ValueError(f"n must be less than or equal to n_.")
        return X, y
