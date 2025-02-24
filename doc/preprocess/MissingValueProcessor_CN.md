# MissingValueProcessor 转化器




## 一、简介

MissingValueProcessor 转化器是用于对缺失值的处理，支持对缺失值进行指定统计量的填补以及差值填补。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名            | 类型      |  默认值   | 描述                                               |
|:----------------|:---------|:------:|:--------------------------------------------------|
| strategy       | {'mean', 'median', 'interpolate'} | 'mean' | 缺失值处理策略                                       |
| interpolate_method | str   | 'linear' | 插值方法（仅当 strategy='interpolate' 时有效）。支持以下方法：- 'linear': 忽略索引并假设值之间距离相等。此方法仅在 MultiIndexes 上支持。- 'time': 在每日或更高分辨率的数据上进行插值，根据给定的时间间隔长度。- 'index', 'values': 使用实际的索引值。- 'pad': 使用现有值填充 NaN。- 'nearest', 'zero', 'slinear', 'quadratic', 'cubic', 'spline', 'barycentric', 'polynomial': 传递给 `scipy.interpolate.interp1d`，这些方法使用实际的索引值。  'polynomial' 和 'spline' 方法需要你指定一个 `order`（int），例如 `df.interpolate(method='polynomial', order=5)`。- 'krogh', 'piecewise_polynomial', 'spline', 'pchip', 'akima', 'cubicspline': 包装了 SciPy 中类似名称的插值方法（'piecewise_polynomial' 在 SciPy 0.18 版本后被 'from_derivatives' 替代）。- 'from_derivatives': 引用 `scipy.interpolate.BPoly.from_derivatives`，这是在 SciPy 0.18 版本后 'piecewise_polynomial' 代替的插值方法。 |




## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

1. 均值填补
   ```python
   import numpy as np
   import pandas as pd
   
   from alpha_feature_tools.preprocess import MissingValueProcessor
   
   data = pd.DataFrame({
       'age': [25, 30, np.nan, 35, 40],
       'income': [50000, np.nan, 70000, 80000, 90000],
       'label': [1, 0, 1, 0, 1]
   })
   X = data[["age", "income"]]
   y = data["label"]
   
   mvp = MissingValueProcessor(strategy="mean")
   mvp.fit(X, y)
   out = mvp.transform(X)
   print(mvp.get_feature_names_out())
   print(out)
   ```
   
   运行结果
   ```txt
   ['age', 'income']
   [[2.50e+01 5.00e+04]
    [3.00e+01 7.25e+04]
    [3.25e+01 7.00e+04]
    [3.50e+01 8.00e+04]
    [4.00e+01 9.00e+04]]
   ```
   
2. 插值填补
   ```python
   import numpy as np
   import pandas as pd
   
   from alpha_feature_tools.preprocess import MissingValueProcessor
   
   data = pd.DataFrame({
       'age': [25, 30, np.nan, 35, 40],
       'income': [50000, np.nan, 70000, 80000, 90000],
       'label': [1, 0, 1, 0, 1]
   })
   X = data[["age", "income"]]
   y = data["label"]
   
   mvp = MissingValueProcessor(strategy="interpolate")
   out = mvp.fit_transform(X)
   print(mvp.get_feature_names_out())
   print(out)
   ```
   
   运行结果
   ```txt
   ['age', 'income']
   [[2.50e+01 5.00e+04]
    [3.00e+01 6.00e+04]
    [3.25e+01 7.00e+04]
    [3.50e+01 8.00e+04]
    [4.00e+01 9.00e+04]]
   ```
