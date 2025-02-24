import pandas as pd

from alpha_feature_tools.preprocess import StandardScaler, MinMaxScaler

data = pd.DataFrame(data=[[1, 223, 0], [2, 653, 0], [9, 100, 1]], columns=["a", "b", "label"])
X = data[["a", "b"]]
y = data["label"]

ss = StandardScaler()
ss.fit(X, y)
out = ss.transform(X)
print(ss.get_feature_names_out())
print(out)

mms = MinMaxScaler()
mms.fit(X, y)
out = mms.transform(X)
print(mms.get_feature_names_out())
print(out)
