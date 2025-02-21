import pandas as pd

from alpha_feature_tools.preprocess import OrdinalEncoder, OneHotEncoder

data = pd.DataFrame(
    data=[
        ['Male', "paris", 0],
        ['Female', "paris", 1],
        ['Female', "tokyo", 1]
    ],
    columns=["sex", "city", "label"]
)
X = data[["sex", "city"]]
y = data["label"]

oe = OrdinalEncoder()
oe.fit(X, y)
print(oe.transform(X))
print(oe.get_feature_names_out())

oh = OneHotEncoder(sparse=False)
oh.fit(X, y)
print(oh.transform(X))
print(oh.get_feature_names_out())
