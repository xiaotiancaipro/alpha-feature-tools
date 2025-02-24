# ChiSquareSelector 转化器




## 一、简介

ChiSquareSelector 转化器是用于计算特征与标签的卡方值以及对应的卡方分布 p 值，通过这两个值进行特征筛选。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名            | 类型      |  默认值   | 描述                                     |
|:-----------------|:---------|:--------:|:---------------------------------------|
| k                | int      | None    | 基于卡方值筛选。如果为 None，则使用 p_threshold 进行选择。 |
| p_threshold      | float    | None    | 基于 p 值筛选。如果为 None，则使用 k 进行选择。          |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。




## 四、示例

```python
import pandas as pd
from sklearn.datasets import load_iris

from alpha_feature_tools.selection import ChiSquareSelector

data = load_iris()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

css = ChiSquareSelector(k=None, p_threshold=0.01)
css.fit(X, y)
print(css.get_feature_names_out())
print(pd.DataFrame(css.get_feature_stats()).T.reset_index(drop=True))
```

运行结果
```txt
['sepal length (cm)', 'petal length (cm)', 'petal width (cm)']
        feature_name  chi2_score   p_value
0  sepal length (cm)   10.817821  0.004477
1   sepal width (cm)    3.710728  0.156396
2  petal length (cm)  116.312613       0.0
3   petal width (cm)    67.04836       0.0
```
