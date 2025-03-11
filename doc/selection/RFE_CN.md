# RFECV 转化器




## 一、简介

RFE 转化器是一种用于特征筛选的方法，用于自动选择最优特征子集以提高模型效果。

该转化器继承自 sklearn.feature_selection.RFE。



## 二、参数说明

| 参数名            | 类型      |  默认值   | 描述                                               |
|:-----------------|:---------|:--------:|:--------------------------------------------------|
| estimator        | ``Estimator`` instance |  | 一个带有 ``fit`` 方法的监督学习估算器，该方法提供了关于特征重要性的信息（例如 `coef_`, `feature_importances_`）。 |
| n_features_to_select | int or float |  `None` | 要选择的特征数量。如果为 `None`，则选择一半的特征。如果为整数，则参数为绝对的特征数。如果为浮点数且在 0 到 1 之间，则参数为特征数的百分比。 |
| step             | int or float | 1 | 如果大于或等于 1，则 ``step`` 对应于每次迭代中要删除的特征数（整数）。如果在 0.0 和 1.0 之间，则 ``step`` 对应于每次迭代中要删除的特征百分比（取整）。 |
| verbose          | int      | 0 | 控制输出的verbosity。 |
| importance_getter | str or callable | 'auto' | 如果为 'auto'，则使用 estimater 的 `coef_` 或 `feature_importances_` 属性用于特征重要性。 <br> 也可以是一个字符串，指定提取特征重要性的属性名称/路径（使用 `attrgetter` 实现）。例如，给 `regressor_.coef_` 在 :class:`~sklearn.compose.TransformedTargetRegressor` 中或 `named_steps.clf.feature_importances_` 在 class:`~sklearn.pipeline.Pipeline` 的最后一个步骤名为 `clf` 中。 <br> 如果为 `callable`，则覆盖默认的重要性和获取器。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
from sklearn.datasets import make_friedman1
from sklearn.svm import SVR

from alpha_feature_tools.selection import RFE

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFE(estimator, n_features_to_select=5, step=1)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
```

运行结果
```txt
[ True  True  True  True  True False False False False False]
[1 1 1 1 1 6 4 3 2 5]
```

