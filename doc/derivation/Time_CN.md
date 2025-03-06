# Time 转化器




## 一、简介

Time 转化器是用于在给定的数据集中将时序特征衍生出相关的时间特征。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

无参数



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
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
```

运行结果
```txt
['Time_feature_0_year', 'Time_feature_0_month', 'Time_feature_0_day', 'Time_feature_1_year', 'Time_feature_1_month', 'Time_feature_1_day']
[[2025    5    6 2025    5    7]
 [2025    5    8 2025    5    9]]
```
