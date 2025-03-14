# SequentialFeatureSelector 转化器




## 一、简介

SequentialFeatureSelector 是一个用于逐步选择特征的转化器，它通过逐步添加或删除特征来构建最优特征子集，通过迭代的方式评估特征的重要性，并根据一定的评估标准来决定添加或删除某个特征，从而优化预测模型的性能。

该转化器继承自 sklearn.feature_selection.SequentialFeatureSelector。



## 二、参数说明

| 参数名            | 类型      | 默认值   | 描述                                                                                                                                                                                                                                                                                                                             |
|:----------------|:---------|:-------:|:-------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| estimator       | estimator instance |      | 未拟合的估计器。                                                                                                                                                                                                                                                                                                                       |
| n_features_to_select | "auto", int or float | 'warn' | 如果为"auto"，行为取决于`tol`参数： <br> - 如果`tol`不为`None`，则特征选择直到得分增加不超过`tol`。<br> - 否则，选择一半的特征。 <br> 如果为整数，参数是绝对的特征选择数量。 <br> 如果为浮点数且在0和1之间，它是要选择的特征的分数。                                                                                                                                                                                 |
| tol             | float    | None   | 如果得分在两次连续的特征添加或删除之间增加不足`tol`，则停止添加或删除。 `tol`仅当`n_features_to_select`为"auto"时启用。                                                                                                                                                                                                                                                |
| direction       | {'forward', 'backward'} | 'forward' | 是否执行向前选择或向后选择。                                                                                                                                                                                                                                                                                                                 |
| scoring         | str, callable, list/tuple or dict | None   | 评估预测在测试集上的单个值（见：ref:`scoring_parameter`）或调用（见：ref:`scoring`）的单个值。<br>注意，当使用自定义评估器时，每个评估器应返回一个值。返回值列表/数组的度量函数可以包装成多个评估器，每个评估器返回一个值。 <br> 如果为`None`，则使用估计器的`score`方法。                                                                                                                                                            |
| cv              | int, cross-validation generator or an iterable | None   | 确定交叉验证拆分策略。<br> 可能的输入为： <br> - 为None，默认使用5折交叉验证 <br> - 整数，指定 `(Stratified)KFold` 中的折叠数量 <br> - 交叉验证生成器 <br> - 生成器生成 (训练, 测试) 切分的数组 <br> 对于整数/None输入，如果估计器是一个分类器且`y`是二元或多元分类，使用 `StratifiedKFold`。在其他情况下，使用 `KFold`。这些生成器实例化为 `shuffle=False`，以便在不同调用中拆分相同。 <br> 参见 :ref:`User Guide <cross_validation>` 获取可以使用在这里的多种交叉验证策略的信息。 |
| n_jobs          | int      | None   | 并行运行的任务数。在评估新特征的添加或删除时，交叉验证过程按折叠并行运行。 `None` 表示1，除非在 :obj:`joblib.parallel_backend` 上下文内。 `-1` 表示使用所有处理器。参见 :term:`Glossary <n_jobs>` 获取更多信息。                                                                                                                                                                                  |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
from alpha_feature_tools.selection import SequentialFeatureSelector
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris

X, y = load_iris(return_X_y=True)
knn = KNeighborsClassifier(n_neighbors=3)
sfs = SequentialFeatureSelector(knn, n_features_to_select=3)
sfs.fit(X, y)
out = sfs.transform(X)
print(sfs.get_support())
print(out.shape)
```

运行结果
```txt
[ True False  True  True]
(150, 3)
```

