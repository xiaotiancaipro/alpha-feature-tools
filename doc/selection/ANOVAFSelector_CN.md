# ANOVAFSelector 转化器




## 一、简介

ANOVAFSelector 转化器是用于计算 **连续型** 特征与标签的 F 值以及对应 F 分布的 p 值，通过这两个值进行特征筛选。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名         | 类型      |  默认值   | 描述                                      |
|:------------|:---------|:--------:|:----------------------------------------|
| f           | float      | None    | 基于 F 值筛选。如果为 None，则使用 p_threshold 进行选择。 |
| p_threshold | float    | None    | 基于 p 值筛选。如果为 None，则使用 f 进行选择。           |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。




## 四、示例

```python
import pandas as pd
from sklearn.datasets import load_iris

from alpha_feature_tools.selection import ANOVAFSelector

data = load_iris()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

css = ANOVAFSelector(f=None, p_threshold=0.01)
css.fit(X, y)
print(css.get_feature_names_out())
print(css.get_feature_stats())
```

运行结果
```txt
['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
{'sepal length (cm)': {'feature_name': 'sepal length (cm)', 'f_value': 119.26450218449871, 'p_value': 1.6696691907731882e-31}, 'sepal width (cm)': {'feature_name': 'sepal width (cm)', 'f_value': 49.16004008961427, 'p_value': 4.492017133303181e-17}, 'petal length (cm)': {'feature_name': 'petal length (cm)', 'f_value': 1180.1611822529776, 'p_value': 2.85677661096213e-91}, 'petal width (cm)': {'feature_name': 'petal width (cm)', 'f_value': 960.0071468018293, 'p_value': 4.169445839437135e-85}}
```
