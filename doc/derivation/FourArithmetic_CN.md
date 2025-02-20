# FourArithmetic 转化器




## 一、简介

FourArithmetic 转化器是用于在给定的数据集中生成通过四则运算后的特征。它能够将两个或多个**连续型特征**通过指定数量的特征进行四则运算衍生出一个或多个新特征。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名        | 类型      | 默认值 | 描述                                                 |
| :------------ | :-------- | :----: | :--------------------------------------------------- |
| n             | int       |   2    | 每次进行四则运算组合的元素个数，默认为双变量四则运算 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

1. 双变量四则运算特征衍生
   ```python
   import pandas as pd
   
   from alpha_feature_tools import derivation
   
   
   # 创建数据
   data = pd.DataFrame({
       "fea_1": [2, 3, 4, 6],
       "fea_2": [1, 1, 3, 4],
       "fea_3": [6, 6, 6, 7]
   })
   
   # 示例化 FourArithmetic 转化器
   fa = derivation.FourArithmetic()
   
   # 拟合并转换数据
   new_features_data = fa.fit_transform(data[["fea_1", "fea_2", "fea_3"]])
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = fa.get_feature_names_out()
   print("new_features_list: ", new_features_list)
   ```
   
   运行结果
   ```txt
   new_features_data: 
       FourArithmetic_fea_1_+_fea_2  FourArithmetic_fea_1_-_fea_2  \
   0                      2.000000                      2.000000   
   1                      3.000000                      3.000000   
   2                      1.333333                      1.333333   
   3                      1.500000                      1.500000   
   
      FourArithmetic_fea_1_*_fea_2  FourArithmetic_fea_1_/_fea_2  \
   0                      2.000000                      2.000000   
   1                      3.000000                      3.000000   
   2                      1.333333                      1.333333   
   3                      1.500000                      1.500000   
   
      FourArithmetic_fea_2_+_fea_3  FourArithmetic_fea_2_-_fea_3  \
   0                      0.166667                      0.166667   
   1                      0.166667                      0.166667   
   2                      0.500000                      0.500000   
   3                      0.571429                      0.571429   
   
      FourArithmetic_fea_2_*_fea_3  FourArithmetic_fea_2_/_fea_3  \
   0                      0.166667                      0.166667   
   1                      0.166667                      0.166667   
   2                      0.500000                      0.500000   
   3                      0.571429                      0.571429   
   
      FourArithmetic_fea_1_+_fea_3  FourArithmetic_fea_1_-_fea_3  \
   0                      0.333333                      0.333333   
   1                      0.500000                      0.500000   
   2                      0.666667                      0.666667   
   3                      0.857143                      0.857143   
   
      FourArithmetic_fea_1_*_fea_3  FourArithmetic_fea_1_/_fea_3  
   0                      0.333333                      0.333333  
   1                      0.500000                      0.500000  
   2                      0.666667                      0.666667  
   3                      0.857143                      0.857143  
   new_features_list:  ['FourArithmetic_fea_1_+_fea_2', 'FourArithmetic_fea_1_-_fea_2', 'FourArithmetic_fea_1_*_fea_2', 'FourArithmetic_fea_1_/_fea_2', 'FourArithmetic_fea_2_+_fea_3', 'FourArithmetic_fea_2_-_fea_3', 'FourArithmetic_fea_2_*_fea_3', 'FourArithmetic_fea_2_/_fea_3', 'FourArithmetic_fea_1_+_fea_3', 'FourArithmetic_fea_1_-_fea_3', 'FourArithmetic_fea_1_*_fea_3', 'FourArithmetic_fea_1_/_fea_3']
   ```
   
2. 三变量四则运算特征衍生
   ```python
   import pandas as pd
   
   from alpha_feature_tools import derivation
   
   
   # 创建数据
   data = pd.DataFrame({
       "fea_1": [2, 3, 4, 6],
       "fea_2": [1, 1, 3, 4],
       "fea_3": [6, 6, 6, 7]
   })
   
   # 示例化 FourArithmetic 转化器
   fa = derivation.FourArithmetic(n=3)
   
   # 拟合并转换数据
   new_features_data = fa.fit_transform(data[["fea_1", "fea_2", "fea_3"]])
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = fa.get_feature_names_out()
   print("new_features_list: ", new_features_list)
   ```
   
   运行结果
   ```txt
   new_features_data: 
       FourArithmetic_fea_1_+_fea_2_+_fea_3  FourArithmetic_fea_1_-_fea_2_-_fea_3  \
   0                              0.333333                              0.333333   
   1                              0.500000                              0.500000   
   2                              0.222222                              0.222222   
   3                              0.214286                              0.214286   
   
      FourArithmetic_fea_1_*_fea_2_*_fea_3  FourArithmetic_fea_1_/_fea_2_/_fea_3  
   0                              0.333333                              0.333333  
   1                              0.500000                              0.500000  
   2                              0.222222                              0.222222  
   3                              0.214286                              0.214286  
   new_features_list:  ['FourArithmetic_fea_1_+_fea_2_+_fea_3', 'FourArithmetic_fea_1_-_fea_2_-_fea_3', 'FourArithmetic_fea_1_*_fea_2_*_fea_3', 'FourArithmetic_fea_1_/_fea_2_/_fea_3']
   ```

