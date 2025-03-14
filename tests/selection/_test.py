from sklearn.linear_model import LogisticRegression

from alpha_feature_tools.selection import SelectFromModel

X = [
    [0.87, -1.34, 0.31],
    [-2.79, -0.02, -0.85],
    [-1.34, -0.48, -2.55],
    [1.92, 1.48, 0.65]
]
y = [0, 1, 0, 1]
selector = SelectFromModel(estimator=LogisticRegression())
selector.fit(X, y)
out = selector.transform(X)
print(selector.estimator_.coef_)
print(selector.threshold_)
print(selector.get_support())
print(out)
