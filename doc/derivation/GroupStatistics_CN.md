# GroupStatistics 转化器




## 一、简介

GroupStatistics 转化器是用于在给定的数据集中分组统计生成新的特征。它能够将一个或多个**连续型**特征通过一个或多个**离散型**特征进行分组统计衍生出多个新特征。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名         | 类型            | 默认值 | 描述                                                         |
| :------------- | :-------------- | :----: | :----------------------------------------------------------- |
| group_features | List[List[str]] |  必填  | 指定分组统计的离散型特征列表，其中嵌套的列表可以组成进行分组统计 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

1. 单变量连续型特征通过单变量离散特征分组统计
   ```python
   import pandas as pd
   from alpha_feature_tools.derivation import GroupStatistics
   
   X = pd.DataFrame({
       "a": [1, 2, 7, 4, 5, 6],
       "b": [1, 1, 1, 1, 2, 2],
   })
   gs = GroupStatistics(group_features=[["b"]])
   gs.fit(X)
   out = gs.transform(X)
   print(gs.get_feature_names_out())
   print(out)
   ```
   
   运行结果
   ```txt
   ['GroupStatistics_a_by_b_grouped_mean', 'GroupStatistics_a_by_b_grouped_median']
   [[3.5 3. ]
    [3.5 3. ]
    [3.5 3. ]
    [3.5 3. ]
    [5.5 5.5]
    [5.5 5.5]]
   ```

2. 单变量连续型特征通过多变量离散特征分组统计
   ```python
   import pandas as pd
   from alpha_feature_tools.derivation import GroupStatistics
   
   X = pd.DataFrame({
       "a": [1, 2, 7, 4, 5, 6],
       "b": [1, 1, 1, 1, 2, 2],
       "c": [1, 1, 2, 2, 2, 2]
   })
   gs = GroupStatistics(group_features=[["b"], ["c"]])
   gs.fit(X)
   out = gs.transform(X)
   print(gs.get_feature_names_out())
   print(out)
   ```
   
   运行结果
   ```txt
   ['GroupStatistics_a_by_b_grouped_mean', 'GroupStatistics_a_by_b_grouped_median', 'GroupStatistics_a_by_c_grouped_mean', 'GroupStatistics_a_by_c_grouped_median']
   [[3.5 3.  1.5 1.5]
    [3.5 3.  1.5 1.5]
    [3.5 3.  5.5 5.5]
    [3.5 3.  5.5 5.5]
    [5.5 5.5 5.5 5.5]
    [5.5 5.5 5.5 5.5]]
   ```
   
3. 单变量连续型特征通过多变量离散特征组合分组统计
   ```python
   import pandas as pd
   from alpha_feature_tools.derivation import GroupStatistics
   
   X = pd.DataFrame({
       "a": [1, 2, 7, 4, 5, 6],
       "b": [1, 1, 1, 1, 2, 2],
       "c": [1, 1, 2, 2, 2, 2]
   })
   gs = GroupStatistics(group_features=[["b", "c"]])
   gs.fit(X)
   out = gs.transform(X)
   print(gs.get_feature_names_out())
   print(out)
   ```
   
   运行结果
   ```txt
   ['GroupStatistics_a_by_b&c_grouped_mean', 'GroupStatistics_a_by_b&c_grouped_median']
   [[1.5 1.5]
    [1.5 1.5]
    [5.5 5.5]
    [5.5 5.5]
    [5.5 5.5]
    [5.5 5.5]]
   ```

4. 多变量连续型特征通过单变量离散特征分组统计
   ```python
   import pandas as pd
   from alpha_feature_tools.derivation import GroupStatistics
   
   X = pd.DataFrame({
       "a": [1, 2, 7, 4, 5, 6],
       "b": [1, 1, 1, 1, 2, 2],
       "c": [1, 2, 3, 4, 5, 6],
       "d": [1, 1, 2, 2, 2, 2]
   })
   gs = GroupStatistics(group_features=[["b"]])
   gs.fit(X)
   out = gs.transform(X)
   print(gs.get_feature_names_out())
   print(out)
   
   ```

   运行结果
   ```txt
   ['GroupStatistics_d_by_b_grouped_mean', 'GroupStatistics_d_by_b_grouped_median', 'GroupStatistics_a_by_b_grouped_mean', 'GroupStatistics_a_by_b_grouped_median', 'GroupStatistics_c_by_b_grouped_mean', 'GroupStatistics_c_by_b_grouped_median']
   [[1.5 1.5 3.5 3.  2.5 2.5]
    [1.5 1.5 3.5 3.  2.5 2.5]
    [1.5 1.5 3.5 3.  2.5 2.5]
    [1.5 1.5 3.5 3.  2.5 2.5]
    [2.  2.  5.5 5.5 5.5 5.5]
    [2.  2.  5.5 5.5 5.5 5.5]]
   ```
