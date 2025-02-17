from _base import data_example
from alpha_feature_tools.derivation import BinaryCrossCombination

data = data_example()

derivation = BinaryCrossCombination(["occupation", "education", "gender"], is_one_hot=False)
out = derivation.fit_transform(data)
print(out)

derivation = BinaryCrossCombination(["occupation", "education", "gender"], is_one_hot=True)
derivation.fit(data)
out = derivation.transform(data)
print(out)
