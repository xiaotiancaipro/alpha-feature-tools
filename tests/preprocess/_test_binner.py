import pandas as pd

from alpha_feature_tools.preprocess import FeatureBinner

data = pd.DataFrame({
    "fea_1": [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 6, 1],
    "fea_2": range(15),
    "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
})
X = data[["fea_1", "fea_2"]]
y = data["target"]

binner = FeatureBinner("equal_width")
binner.fit(X)
out = binner.transform(X)
print(out)

binner = FeatureBinner(method="equal_freq")
binner.fit(X)
out = binner.transform(X)
print(out)
