from typing import List

import pandas as pd
from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

from .._base import _BaseTransformer


class MutualInfoSelector(_BaseTransformer):
    """
    Mutual information feature selector

    Parameters
    ----------
    problem_type : str, default='classification'
        The type of problem to solve. Must be 'classification' or 'regression'.

    k : int, default=None
        The number of top features to select. If None, the threshold is used.

    mi_threshold : float, default=None
        The threshold for mutual information. If None, the k is used.

    discrete_features : str or List[bool], default='auto'
        Whether to treat features as discrete or continuous. If 'auto', the feature type is inferred from the data.
        If a list of bools is provided, it must be the same length as the number of features and specify whether each
        feature is discrete or continuous.

    n_neighbors : int, default=3
        The number of neighbors to use for continuous features. Only used if discrete_features is 'auto'.

    random_state : int, default=None
        The random state to use for the feature selection process. If None, the random state is not set.
    """

    def __init__(
            self,
            *,
            problem_type: str = 'classification',
            k: int | None = None,
            mi_threshold: float | None = None,
            discrete_features: str | List[bool] = 'auto',
            n_neighbors: int = 3,
            random_state: int | None = None
    ):
        self.problem_type = problem_type
        self.k = k
        self.mi_threshold = mi_threshold
        self.discrete_features = discrete_features
        self.n_neighbors = n_neighbors
        self.random_state = random_state
        self.__mi_scores = list()
        self.__feature_stats = dict()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:

        if self.problem_type == 'classification':
            self.__mi_scores = mutual_info_classif(
                X=X,
                y=y,
                discrete_features=self.discrete_features,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )

        if self.problem_type == 'regression':
            self.__mi_scores = mutual_info_regression(
                X=X,
                y=y,
                discrete_features=self.discrete_features,
                n_neighbors=self.n_neighbors,
                random_state=self.random_state
            )

        self.__mi_scores = list(self.__mi_scores)
        self.__feature_stats = {fea: mi for fea, mi in zip(self.feature_names_in_, self.__mi_scores)}

        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:

        if self.k is not None:
            temp_dict = {mi_score: index for index, mi_score in enumerate(self.__mi_scores)}
            mi_scores_sorted = sorted(self.__mi_scores, reverse=True)
            feature_names_out_sorted = [self.feature_names_in_[temp_dict[mi_score]] for mi_score in mi_scores_sorted]
            self._feature_names_out = feature_names_out_sorted[:self.k]

        if self.mi_threshold is not None:
            for feature, mi_score in zip(self.feature_names_in_, self.__mi_scores):
                if mi_score >= self.mi_threshold:
                    self._feature_names_out.append(feature)

        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if self.problem_type not in ['classification', 'regression']:
            raise ValueError("The problem_type must be 'classification' or 'regression'")
        if (self.k is None) and (self.mi_threshold is None):
            raise ValueError("Either k or mi_threshold must be specified")
        if (self.k is not None) and (self.mi_threshold is not None):
            raise ValueError("Only one of k or mi_threshold must be specified")
        return X, y

    def _more_tags(self):
        return super()._more_tags().update({"requires_y": True})

    def set_k(self, k: int | None) -> None:
        self.k = k
        return None

    def set_mi_threshold(self, mi_threshold: float | None) -> None:
        self.mi_threshold = mi_threshold
        return None

    def get_feature_stats(self) -> dict:
        return self.__feature_stats
