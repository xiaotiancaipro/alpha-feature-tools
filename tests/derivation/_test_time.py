from alpha_feature_tools.derivation import Time

X = [
    ["2025-05-06", "2025-05-07"],
    ["2025-05-08", "2025-05-09"]
]

time = Time()
time.fit(X)
out = time.transform(X)
print(time.get_feature_names_out())
print(out)
