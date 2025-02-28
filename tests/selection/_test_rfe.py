from sklearn.datasets import make_friedman1
from sklearn.svm import SVR

from alpha_feature_tools.selection import RFECV

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
