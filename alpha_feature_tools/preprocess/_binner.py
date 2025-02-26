import numpy as np
import pandas as pd

from .._base import _BaseTransformer


class FeatureBinner(_BaseTransformer):

    def __init__(self, method="equal_width", n_bins=10):
        """
        Custom Feature Binning Transformer

        Parameters
        ----------
        method : str, default='equal_width'
            Binning method, options are 'equal_width', 'equal_freq', 'kmeans', 'woe'

        n_bins : int, default=10
            Number of bins for 'equal_width' and 'equal_freq' methods
        """
        self.method = method
        self.n_bins = n_bins
        self.__epsilon = 1e-6  # 平滑系数，用于WOE计算时避免除以零
        self.__bin_edges = dict()
        self.__woe_dict = dict()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for feature in self.feature_names_in_:
            edges = list()
            if self.method == "equal_width":
                edges = np.linspace(np.min(X[feature]), np.max(X[feature]), self.n_bins + 1)
                edges = np.unique(edges)
            if self.method == "equal_freq":
                quantiles = np.linspace(0, 1, self.n_bins + 1)
                edges = np.quantile(X[feature], quantiles)
                edges = np.unique(edges)
                edges[0] = -np.inf
                edges[-1] = np.inf
            # if self.method == "kmeans":
            #     kmeans = KMeans(n_clusters=self.n_bins, random_state=0)
            #     print("=====")
            #     print(np.array(X[feature].tolist()).reshape(1, -1))
            #     kmeans.fit(np.array(X[feature].tolist()).reshape(1, -1))
            #     centers = np.sort(kmeans.cluster_centers_.flatten())
            #     boundaries = (centers[:-1] + centers[1:]) / 2
            #     edges = np.concatenate([[-np.inf], boundaries, [np.inf]])
            # if self.method == "woe":
            #     quantiles = np.linspace(0, 1, self.n_bins + 1)
            #     edges = np.quantile(X[feature], quantiles)
            #     edges = np.unique(edges)
            #     edges[0] = -np.inf
            #     edges[-1] = np.inf
            #     binned = pd.cut(X[feature], bins=edges, labels=False, include_lowest=True)
            #     df = pd.DataFrame({"feature": binned, "target": y})
            #     grouped = df.groupby("feature")["target"].agg([
            #         ("bad", lambda x: (x == 1).sum() + self.__epsilon),
            #         ("good", lambda x: (x == 0).sum() + self.__epsilon)
            #     ])
            #     total_bad = df["target"].sum() + 2 * self.__epsilon
            #     total_good = len(df) - df["target"].sum() + 2 * self.__epsilon
            #     grouped["bad_pct"] = grouped["bad"] / total_bad
            #     grouped["good_pct"] = grouped["good"] / total_good
            #     grouped["woe"] = np.log(grouped["bad_pct"] / grouped["good_pct"])
            #     self.woe_dict[feature] = grouped["woe"].to_dict()
            self.__bin_edges[feature] = edges
            self._feature_names_out.append(feature)
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature, bin_edges in self.__bin_edges.items():
            # if self.method == "woe":
            #     woe_map = self.woe_dict[feature]
            #     woe_values = np.array([woe_map.get(x, 0) for x in bin_edges])
            #     print("stop")
            X[feature] = pd.cut(X[feature], bins=bin_edges, labels=False, include_lowest=True)
        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        if self.method not in ["equal_width", "equal_freq", "kmeans", "woe"]:
            raise ValueError("The method must be 'equal_width', 'equal_freq', 'kmeans' or 'woe'")
        if self.method == "woe" and y is None:
            raise ValueError("WOE binner requires target values for training")
        return X, y
