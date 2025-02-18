from _base import data_example
from alpha_feature_tools.derivation import FourArithmetic

data = data_example()

derivation = FourArithmetic(["fea_5", "fea_6"])
out = derivation.fit_transform(data)
print(out)

derivation = FourArithmetic(["fea_5", "fea_6", "fea_7"], n=3)
out = derivation.fit_transform(data)
print(out)
