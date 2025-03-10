import pandas as pd

from alpha_feature_tools.derivation import NLP

X = pd.DataFrame(
    data=[
        [0, 1, 0, 2, 3],
        [3, 2, 1, 0, 0]
    ],
    columns=["a", "b", "c", "d", "e"]
)

nlp = NLP(is_tf_idf=False)
nlp.fit(X)
out = nlp.transform(X)
print(nlp.get_feature_names_out())
print(out)

nlp = NLP(is_tf_idf=True)
nlp.fit(X)
out = nlp.transform(X)
print(nlp.get_feature_names_out())
print(out)
