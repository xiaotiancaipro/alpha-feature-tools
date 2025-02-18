from _base import data_example
from alpha_feature_tools.derivation import CrossCombination

data = data_example()

derivation = CrossCombination(["fea_1", "fea_2", "fea_3"])
out = derivation.fit_transform(data)
print(out)

derivation = CrossCombination(feature_names=["fea_1", "fea_2", "fea_3"], is_one_hot=True)
derivation.fit(data)
out = derivation.transform(data)
print(out)
