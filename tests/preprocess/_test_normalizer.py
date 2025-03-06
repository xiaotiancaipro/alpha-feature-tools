from alpha_feature_tools.preprocess import Normalizer

X = [
    [4, 1, 2, 2],
    [1, 3, 9, 3],
    [5, 7, 5, 1]
]
transformer = Normalizer()
transformer.fit(X)
out = transformer.transform(X)
print(out)
