# RFECV 转化器




## 一、简介

RFECV 转化器是一种用于特征筛选的方法，用于自动选择最优特征子集以提高模型效果。

该转化器继承自 sklearn.feature_selection.RFECV。



## 二、参数说明

| 参数名            | 类型      | 默认值 | 描述                                               |
|:-----------------|:---------|:------:|:------------------------------------------------- |
| estimator        | ``Estimator`` instance | -    | 一个带有 ``fit`` 方法的监督学习估计器，该方法提供有关特征重要性的信息，通过 ``coef_`` 属性或 ``feature_importances_`` 属性。 |
| step             | int or float | 1 | 如果大于或等于 1，则 ``step`` 对应于每次迭代要移除的整数特征数。如果在 (0.0, 1.0) 之间，则 ``step`` 对应于每次迭代要移除的特征百分比（向下舍入）。注意，最后一轮迭代可能会移除比 ``step`` 更少的特征以达到 ``min_features_to_select``。 |
| min_features_to_select | int | 1 | 要选择的最小特征数。这个数量的特征将始终被评分，即使原始特征数量与 ``min_features_to_select`` 之间的差异不能被 ``step`` 整除。 |
| cv               | int, cross-validation generator or an iterable | None | 确定交叉验证拆分策略。可能的输入包括：- None，使用默认的 5 折交叉验证，- 整数，指定折数。- :term:`CV splitter`，- 生成（训练，测试）拆分的迭代器。 对于整数/None 输入，如果 ``y`` 为二元或多分类，则使用 :class:`~sklearn.model_selection.StratifiedKFold`。如果估计器是分类器或 ``y`` 既不是二元也不是多分类，则使用 :class:`~sklearn.model_selection.KFold`。参考 :ref:`User Guide <cross_validation>` 了解可以在此处使用的各种交叉验证策略。 |
| scoring          | str, callable or None | None | 一个字符串 (参见模型评估文档) 或一个评分可调用对象/函数，具有签名 ``scorer(estimator, X, y)``。 |
| verbose          | int | 0 | 控制输出的响度。 |
| n_jobs           | int or None | None | 在交叉验证各折的拟合过程中并行运行的核数。``None`` 表示 1 个核，除非处于 :obj:`joblib.parallel_backend` 上下文中，此时表示所有处理器。``-1`` 表示使用所有处理器。参见 :term:`Glossary <n_jobs>` 了解更多信息。 |
| importance_getter | str or callable | 'auto' | 如果为 'auto'，使用 ``coef_`` 或 ``feature_importances_`` 属性获取特征重要性。同时也接受一个字符串，指定属性名称/路径以提取特征重要性，例如对于 :class:`~sklearn.compose.TransformedTargetRegressor`，可以是 `regressor_.coef_`；对于 :class:`~sklearn.pipeline.Pipeline` 且最后一步命名为 `clf` 时，可以是 `named_steps.clf.feature_importances_`。如果为 `callable`，则会覆盖默认的特征重要性获取器。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
from sklearn.datasets import make_friedman1
from sklearn.svm import SVR

from alpha_feature_tools.selection import RFECV

X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
estimator = SVR(kernel="linear")
selector = RFECV(estimator, step=1, cv=5)
selector = selector.fit(X, y)
print(selector.support_)
print(selector.ranking_)
```

运行结果
```txt
[ True  True  True  True  True False False False False False]
[1 1 1 1 1 6 4 3 2 5]
```

