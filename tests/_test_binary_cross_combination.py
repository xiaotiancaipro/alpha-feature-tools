from _base import data_example
from alpha_feature_tools.derivation import CrossCombination

data = data_example()

derivation = CrossCombination(["occupation", "education", "gender"])
out = derivation.fit_transform(data)
print(out)

derivation = CrossCombination(["occupation", "education", "gender"], is_one_hot=True)
derivation.fit(data)
out = derivation.transform(data)
print(out)
