import pandas as pd
from alpha_feature_tools.derivation import GroupStatistics

X = pd.DataFrame({
    "a": [1, 2, 7, 4, 5, 6],
    "b": [1, 1, 1, 1, 2, 2],
})
gs = GroupStatistics(group_features=[["b"]])
gs.fit(X)
out = gs.transform(X)
print(gs.get_feature_names_out())
print(out)

X = pd.DataFrame({
    "a": [1, 2, 7, 4, 5, 6],
    "b": [1, 1, 1, 1, 2, 2],
    "c": [1, 1, 2, 2, 2, 2]
})
gs = GroupStatistics(group_features=[["b"], ["c"]])
gs.fit(X)
out = gs.transform(X)
print(gs.get_feature_names_out())
print(out)

X = pd.DataFrame({
    "a": [1, 2, 7, 4, 5, 6],
    "b": [1, 1, 1, 1, 2, 2],
    "c": [1, 1, 2, 2, 2, 2]
})
gs = GroupStatistics(group_features=[["b", "c"]])
gs.fit(X)
out = gs.transform(X)
print(gs.get_feature_names_out())
print(out)

X = pd.DataFrame({
    "a": [1, 2, 7, 4, 5, 6],
    "b": [1, 1, 1, 1, 2, 2],
    "c": [1, 2, 3, 4, 5, 6],
    "d": [1, 1, 2, 2, 2, 2]
})
gs = GroupStatistics(group_features=[["b"]])
gs.fit(X)
out = gs.transform(X)
print(gs.get_feature_names_out())
print(out)
