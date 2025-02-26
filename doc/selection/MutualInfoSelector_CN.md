# MutualInfoSelector 转化器




## 一、简介

MutualInfoSelector 转化器是用于计算指定问题的 **离散型** 特征或 **连续型** 特征的 MI 值，通过该值或该值排序后进行筛选。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名            | 类型      |  默认值   | 描述                                               |
|:----------------|:---------|:--------:|:--------------------------------------------------|
| problem_type    | str      |  'classification' | 解决的问题类型，必须是'classification'或'regression' |
| k              | int      |  None   | 特征数量的阈值。如果为None，则使用互信息的阈值 |
| mi_threshold   | float    |  None   | 互信息的阈值。如果为None，则使用k值                |
| discrete_features | str 或 List[bool]|  'auto'  | 特征是否为离散或连续的标志。如果为'auto'，则从数据中推断特征类型。如果提供了一个包含布尔值的列表，则必须与特征数量相同，并指定每个特征是离散还是连续 |
| n_neighbors    | int      |  3      | 对于连续特征使用的邻居数量，仅在discrete_features为'auto'时使用 |
| random_state   | int      |  None   | 特征选择过程中使用的随机状态，如果为None，则不设置随机状态 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。




## 四、示例

```python
import pandas as pd
from sklearn.datasets import load_iris

from alpha_feature_tools.selection import MutualInfoSelector

data = load_iris()
X, y = pd.DataFrame(data.data, columns=data.feature_names), data.target

ms = MutualInfoSelector(problem_type='classification', k=2)
ms.fit(X, y)
out = ms.transform(X)
print(pd.DataFrame(out, columns=ms.get_feature_names_out()).head())
```

运行结果
```txt
   petal length (cm)  petal width (cm)
0                1.4               0.2
1                1.4               0.2
2                1.3               0.2
3                1.5               0.2
4                1.4               0.2
```
