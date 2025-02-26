# FeatureBinner 转化器




## 一、简介

FeatureBinner 转化器是用于特征分箱。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名            | 类型      | 默认值   | 描述                                    |
|:---------------|:--------|:------:|:--------------------------------------|
| method          | str     | 'equal_width' | 分箱方法，可选值为 'equal_width', 'equal_freq' |
| n_bins          | int     | 10     | 分箱的数量                                 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

1. 等宽分箱
   ```python
   import pandas as pd
   
   from alpha_feature_tools.preprocess import FeatureBinner
   
   data = pd.DataFrame({
       "fea_1": [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 6, 1],
       "fea_2": range(15),
       "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
   })
   X = data[["fea_1", "fea_2"]]
   y = data["target"]
   
   binner = FeatureBinner("equal_width")
   binner.fit(X)
   out = binner.transform(X)
   print(out)
   ```

   运行结果
   ```txt
   [[0 0]
    [0 0]
    [1 1]
    [1 2]
    [2 2]
    [2 3]
    [3 4]
    [4 4]
    [4 5]
    [6 6]
    [7 7]
    [8 7]
    [9 8]
    [6 9]
    [0 9]]
   ```

2. 等频分箱
   ```python
   import pandas as pd
   
   from alpha_feature_tools.preprocess import FeatureBinner
   
   data = pd.DataFrame({
       "fea_1": [1, 1, 2, 2, 3, 3, 4, 5, 5, 6, 7, 8, 9, 6, 1],
       "fea_2": range(15),
       "target": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0]
   })
   X = data[["fea_1", "fea_2"]]
   y = data["target"]
   
   binner = FeatureBinner(method="equal_freq")
   binner.fit(X)
   out = binner.transform(X)
   print(out)
   ```

   运行结果

   ```txt
   [[0 0]
    [0 0]
    [1 1]
    [1 2]
    [2 2]
    [2 3]
    [3 4]
    [4 4]
    [4 5]
    [6 6]
    [7 7]
    [8 7]
    [8 8]
    [6 9]
    [0 9]]
   ```
