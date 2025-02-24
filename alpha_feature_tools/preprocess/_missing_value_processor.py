import pandas as pd

from .._base import _BaseTransformer


class MissingValueProcessor(_BaseTransformer):
    """
    Missing value processing transformer, supporting mean filling, median filling and interpolation filling

    Parameters
    ----------
    strategy : {'mean', 'median', 'interpolate'}, default='mean'
        Missing value processing strategy

    interpolate_method : str, default='linear'
        Interpolation method (only valid when strategy='interpolate'). One of:
        * 'linear': Ignore the index and treat the values as equally
          spaced. This is the only method supported on MultiIndexes.
        * 'time': Works on daily and higher resolution data to interpolate
          given length of interval.
        * 'index', 'values': use the actual numerical values of the index.
        * 'pad': Fill in NaNs using existing values.
        * 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline',
          'barycentric', 'polynomial': Passed to
          `scipy.interpolate.interp1d`. These methods use the numerical
          values of the index.  Both 'polynomial' and 'spline' require that
          you also specify an `order` (int), e.g.
          ``df.interpolate(method='polynomial', order=5)``.
        * 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima',
          'cubicspline': Wrappers around the SciPy interpolation methods of
          similar names. See `Notes`.
        * 'from_derivatives': Refers to
          `scipy.interpolate.BPoly.from_derivatives` which
          replaces 'piecewise_polynomial' interpolation method in
          scipy 0.18.
    """

    def __init__(self, strategy="mean", interpolate_method="linear"):
        self.strategy = strategy
        self.interpolate_method = interpolate_method
        self.__statistics = ["mean", "median"]
        self.__impute_values = dict()
        super().__init__()

    def _fit(self, X: pd.DataFrame, y=None) -> None:
        self._feature_names_out = self.feature_names_in_
        if self.strategy not in self.__statistics:
            return None
        for feature in self.feature_names_in_:
            self.__impute_values[feature] = eval(f"X[feature].{self.strategy}()")
        return None

    def _transform(self, X: pd.DataFrame) -> pd.DataFrame:
        for feature in self.feature_names_in_:
            col = X[feature]
            if self.strategy not in self.__statistics:
                X[feature] = col.interpolate(method=self.interpolate_method)
                continue
            X[feature] = col.fillna(self.__impute_values[feature])
        return X[self.feature_names_in_]

    def _validate_keywords(self, X, y=None) -> tuple:
        X, y = super()._validate_keywords(X, y)
        valid_strategies = [*self.__statistics, "interpolate"]
        if self.strategy not in valid_strategies:
            raise ValueError(
                f"You entered an invalid strategy: {self.strategy}, "
                f"please enter a valid value: {valid_strategies}."
            )
        return X, y
