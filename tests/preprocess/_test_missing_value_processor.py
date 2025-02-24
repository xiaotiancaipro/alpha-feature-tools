import numpy as np
import pandas as pd

from alpha_feature_tools.preprocess import MissingValueProcessor

data = pd.DataFrame({
    'age': [25, 30, np.nan, 35, 40],
    'income': [50000, np.nan, 70000, 80000, 90000],
    'label': [1, 0, 1, 0, 1]
})
X = data[["age", "income"]]
y = data["label"]

mvp = MissingValueProcessor(strategy="mean")
mvp.fit(X, y)
out = mvp.transform(X)
print(mvp.get_feature_names_out())
print(out)

out = mvp.transform([[np.nan, np.nan]])
print(out)

mvp = MissingValueProcessor(strategy="interpolate")
out = mvp.fit_transform(X)
print(mvp.get_feature_names_out())
print(out)
