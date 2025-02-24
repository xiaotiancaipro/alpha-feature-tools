import pandas as pd

from alpha_feature_tools.preprocess import OutlierProcessor

data = pd.DataFrame({
    'age': [25, 30, 45, 35, 150],
    'income': [50000, 60000, 7000000000, 80000, 90000],
    'label': [1, 0, 1, 0, 1]
})
X = data[["age", "income"]]
y = data["label"]

op = OutlierProcessor(method="box")
op.fit(X, y)
out = op.transform(X)
print(op.get_feature_names_out())
print(out)
