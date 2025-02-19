from tests._base import data_example
from alpha_feature_tools.derivation import CrossCombination

data = data_example()

derivation = CrossCombination()
out = derivation.fit_transform(data[["fea_1", "fea_2", "fea_3"]])
print(out)

derivation = CrossCombination(is_one_hot=True)
derivation.fit(data[["fea_1", "fea_2", "fea_3"]])
out = derivation.transform(data)
print(out)
