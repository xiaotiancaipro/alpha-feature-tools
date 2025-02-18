from alpha_feature_tools.preprocess import LabelEncoder, OneHotEncoder

le = LabelEncoder()
le.fit(["paris", "paris", "tokyo", "amsterdam"])
print(list(le.classes_))
le.transform(["tokyo", "tokyo", "paris"])
print(list(le.inverse_transform([2, 2, 1])))
out = le.fit_transform(["tokyo", "tokyo", "paris"])
print(out)

ohe = OneHotEncoder(sparse=False)
out = ohe.fit_transform([['Male', 1], ['Female', 3], ['Female', 2]])
print(out)
print(ohe.categories_)

