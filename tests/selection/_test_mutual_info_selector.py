import pandas as pd
from sklearn.datasets import load_iris

from alpha_feature_tools.selection import MutualInfoSelector

data = load_iris()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

ms = MutualInfoSelector(problem_type='classification', k=2)
ms.fit(X, y)
out = ms.transform(X)
print(pd.DataFrame(out, columns=ms.get_feature_names_out()).head())
