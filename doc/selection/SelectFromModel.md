# SelectFromModel 转化器




## 一、简介

SelectFromModel 转化器是一种基于设定阈值筛选重要性高于该值的特征来实现特征选择的方法。

该转化器继承自 sklearn.feature_selection.SelectFromModel。



## 二、参数说明

| 参数名            | 类型      | 默认值 | 描述                                               |
|:------------------|:----------|:------:|:-------------------------------------------------- |
| estimator         | object    | False | 基础 estimater，从其构建变换器。此基础 estimater 可以是拟合的（如果 `prefit` 设为 True），也可以是非拟合的。拟合后的 estimater 应具有 `feature_importances_` 或 `coef_` 属性，否则应使用 `importance_getter` 参数。 |
| threshold         | str或float | None  | 特征选择时使用的阈值。绝对重要性值大于或等于该阈值的特征保留，其余则舍弃。如果为 "median" 或 "mean"，则阈值为重要性值的中位数或均值。也可以使用缩放因子（例如，"1.25*mean"）。若为 None 且 estimater 的惩罚参数设置为 l1（显式或隐式），则使用 1e-5 作为阈值。否则，默认使用 "mean"。 |
| prefit            | bool     | False | 是否期望直接传递一个已拟合的 estimater 到构造函数中。如果为 True，则 `estimator` 必须是已拟合的 estimater。如果为 False，则将在调用 `fit` 和 `partial_fit` 后拟合和更新 `estimator`。 |
| norm_order        | 非零整数、inf、-inf | 1 | 在 `coef_` 属性维度为 2 时，用于过滤系数向量的范数的阶数。 |
| max_features      | 整数、调用函数、None | None  | 选择的最大特征数。 - 如果是整数，则指定最大特征数。 - 如果是调用函数，那么用 `max_features(X)` 的输出来计算允许的最大特征数。 - 如果为 None，则选择所有特征。要仅基于 `max_features` 选择特征，请将 `threshold` 设为 -np.inf。 |
| importance_getter | 字符串或调用函数、'auto' | 'auto' | 如果是 'auto'，则使用特征重要性通过 `coef_` 属性或 `feature_importances_` 属性来获取。也可以提供一个字符串，指定提取特征重要性的属性名称/路径（通过 `attrgetter` 实现）。例如，在 `TransformedTargetRegressor` 中提供 `regressor_.coef_`，在 `Pipeline` 中最后一步命名为 `clf` 时提供 `named_steps.clf.feature_importances_`。如果是一个调用函数，则会覆盖默认的特征重要性获取器。调用函数会传入拟合后的 estimater 并返回每个特征的重要性。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
from sklearn.linear_model import LogisticRegression

from alpha_feature_tools.selection import SelectFromModel

X = [
    [0.87, -1.34, 0.31],
    [-2.79, -0.02, -0.85],
    [-1.34, -0.48, -2.55],
    [1.92, 1.48, 0.65]
]
y = [0, 1, 0, 1]
selector = SelectFromModel(estimator=LogisticRegression())
selector.fit(X, y)
out = selector.transform(X)
print(selector.estimator_.coef_)
print(selector.threshold_)
print(selector.get_support())
print(out)
```

运行结果
```txt
[[-0.3252302   0.83462377  0.49750423]]
0.5524527319086916
[False  True False]
[[-1.34]
 [-0.02]
 [-0.48]
 [ 1.48]]
```

