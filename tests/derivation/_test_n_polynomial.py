import pandas as pd
from alpha_feature_tools.derivation import NPolynomial

X = pd.DataFrame(
    data=[
        [1, 2, 3, 4, 5],
        [2, 3, 4, 5, 6]
    ],
    columns=["a", "b", "c", "d", "e"]
)

n_ploy = NPolynomial(n=2, n_=2)
n_ploy.fit(X)
print(sorted(n_ploy.get_feature_names_out()))
