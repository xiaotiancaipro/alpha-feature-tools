# OutlierProcessor 转化器




## 一、简介

OutlierProcessor 转化器是用于对异常值的检测和处理，支持对异常值使用指定的方法进行检测和处理。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明


| 参数名            | 类型      |  默认值   | 描述                                               |
|:-----------------|:---------|:--------:|:-------------------------------------------------- |
| method           | str      | ‘std’ | 异常值检测方法，'std'（3-σ规则）或'box'（IQR方法） |
| treatment        | str      | ‘cap’ | 异常值处理方法，'mean'（均值矫正），'cap'（盖帽法），或'none'（不做处理） |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
import pandas as pd

from alpha_feature_tools.preprocess import OutlierProcessor

data = pd.DataFrame({
    'age': [25, 30, 45, 35, 150],
    'income': [50000, 60000, 7000000000, 80000, 90000],
    'label': [1, 0, 1, 0, 1]
})
X = data[["age", "income"]]
y = data["label"]

op = OutlierProcessor(method="box")
op.fit(X, y)
out = op.transform(X)
print(op.get_feature_names_out())
print(out)
```

运行结果
```txt
['age', 'income']
[[2.50e+01 5.00e+04]
 [3.00e+01 6.00e+04]
 [4.50e+01 1.35e+05]
 [3.50e+01 8.00e+04]
 [6.75e+01 9.00e+04]]
```
