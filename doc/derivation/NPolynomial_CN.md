# NPolynomial 转化器




## 一、简介

NPolynomial 转化器是用于在给定的数据集中生成指定阶数的多项式特征。它能够将两个或多个**连续型特征**通过指定数量的特征进行多项式组合衍生出一个或多个新特征。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名 | 类型      | 默认值 | 描述                           |
|:----| :-------- |:---:|:-----------------------------|
| n   | int       |  1  | 每次进行多项式运算组合的元素个数，默认为单变量多项式运算 |
| n_  | int       |  3  | 多项式运算的阶数                     |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

1. 单变量三阶多项式
   ```python
   import pandas as pd
   
   from alpha_feature_tools.derivation import NPolynomial
   
   # 创建数据
   X = pd.DataFrame(
       data=[
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6]
       ],
       columns=["a", "b", "c", "d", "e"]
   )
   
   # 示例化 NPolynomial 转化器
   n_poly = NPolynomial(n=1, n_=3)
   
   # 拟合并转换数据
   new_features_data = n_poly.fit_transform(X)
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = n_poly.get_feature_names_out()
   print("new_features_list: ", new_features_list)
   ```
   
   运行结果
   ```txt
   new_features_data: 
       NPolynomial_a**3  NPolynomial_b**3  NPolynomial_c**3  NPolynomial_d**3  \
   0                 1                 8                27                64   
   1                 8                27                64               125   
   
      NPolynomial_e**3  
   0               125  
   1               216  
   new_features_list:  ['NPolynomial_a**3', 'NPolynomial_b**3', 'NPolynomial_c**3', 'NPolynomial_d**3', 'NPolynomial_e**3']
   ```
   
2. 双变量二阶多项式
   ```python
   import pandas as pd
   
   from alpha_feature_tools.derivation import NPolynomial
   
   # 创建数据
   X = pd.DataFrame(
       data=[
           [1, 2, 3, 4, 5],
           [2, 3, 4, 5, 6]
       ],
       columns=["a", "b", "c", "d", "e"]
   )
   
   # 示例化 NPolynomial 转化器
   n_poly = NPolynomial(n=2, n_=2)
   
   # 拟合并转换数据
   new_features_data = n_poly.fit_transform(X)
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = n_poly.get_feature_names_out()
   print("new_features_list: ", new_features_list)
   ```
   
   运行结果
   ```txt
   new_features_data: 
       NPolynomial_a**1*b**1  NPolynomial_a**2  NPolynomial_b**2  \
   0                      2                 1                 4   
   1                      6                 4                 9   
   
      NPolynomial_a**1*c**1  NPolynomial_c**2  NPolynomial_a**1*d**1  \
   0                      3                 9                      4   
   1                      8                16                     10   
   
      NPolynomial_d**2  NPolynomial_a**1*e**1  NPolynomial_e**2  \
   0                16                      5                25   
   1                25                     12                36   
   
      NPolynomial_b**1*c**1  NPolynomial_b**1*d**1  NPolynomial_b**1*e**1  \
   0                      6                      8                     10   
   1                     12                     15                     18   
   
      NPolynomial_c**1*d**1  NPolynomial_c**1*e**1  NPolynomial_d**1*e**1  
   0                     12                     15                     20  
   1                     20                     24                     30  
   new_features_list:  ['NPolynomial_a**1*b**1', 'NPolynomial_a**2', 'NPolynomial_b**2', 'NPolynomial_a**1*c**1', 'NPolynomial_c**2', 'NPolynomial_a**1*d**1', 'NPolynomial_d**2', 'NPolynomial_a**1*e**1', 'NPolynomial_e**2', 'NPolynomial_b**1*c**1', 'NPolynomial_b**1*d**1', 'NPolynomial_b**1*e**1', 'NPolynomial_c**1*d**1', 'NPolynomial_c**1*e**1', 'NPolynomial_d**1*e**1']
   ```

