import pandas as pd

from .._base import _BaseTransformer


class Time(_BaseTransformer):

    def __init__(self):
        self.__column_list = ["year", "month", "day"]
        self.__op = list()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        for feature in self.feature_names_in_:
            for column in self.__column_list:
                self._feature_names_out.append(f"{self.__class__.__name__}_{feature}_{column}")
                self.__op.append((feature, column))
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for (feature, column), name in zip(self.__op, self._feature_names_out):
            X[feature] = pd.to_datetime(X[feature])
            X[name] = eval(f"X['{feature}'].dt.{column}")
        return X[self._feature_names_out]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        for index in range(len(self.feature_names_in_)):
            feature = self.feature_names_in_[index]
            X[feature] = pd.to_datetime(X[feature])
            if not isinstance(feature, str):
                X = X.rename(columns={feature: f"feature_{feature}"})
                self.feature_names_in_[index] = f"feature_{feature}"
        return X, y
