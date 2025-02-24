import pandas as pd
from sklearn.datasets import load_iris

from alpha_feature_tools.selection import ANOVAFSelector

data = load_iris()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

css = ANOVAFSelector(f=None, p_threshold=0.01)
css.fit(X, y)
print(css.get_feature_names_out())
print(css.get_feature_stats())
