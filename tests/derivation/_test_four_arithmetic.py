from tests._base import data_example
from alpha_feature_tools.derivation import FourArithmetic

data = data_example()

derivation = FourArithmetic()
out = derivation.fit_transform(data[["fea_5", "fea_6", "fea_7"]])
print(out)

derivation = FourArithmetic(n=3)
out = derivation.fit_transform(data[["fea_5", "fea_6", "fea_7"]])
print(derivation.get_feature_names_out())
print(out)
