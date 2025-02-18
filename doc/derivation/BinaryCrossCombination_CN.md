# CrossCombination 转化器




## 一、简介

CrossCombination 转化器是用于在给定的数据集中生成多元交叉组合特征。它能够将两个或多个**类别型特征**通过指定数量的特征进行交叉组合衍生出一个或多个新特征，并可选择对这些新特征进行 One-Hot 编码。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名        | 类型      | 默认值 | 描述                                       |
| :------------ | :-------- | :----: | :----------------------------------------- |
| feature_names | List[str] |  必填  | 需要参与交叉组合的原始特征名称列表         |
| n             | int       |   2    | 几个特征进行交叉组合，默认为双变量交叉组合 |
| is_one_hot    | bool      |  True  | 是否对生成的新特征进行 One-Hot 编码        |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

1. 双变量交叉组合
   ```python
   import pandas as pd
   
   from alpha_feature_tools import derivation
   
   
   # 创建数据
   data = pd.DataFrame({
       "fea_1": ["Q", "W", "W", "Q"],
       "fea_2": ["A", "B", "A", "C"],
       "fea_3": [1, 1, 3, 4],
       "fea_4": [6, 6, 6, 7]
   })
   
   # 示例化 CrossCombination 转化器
   cc = derivation.CrossCombination(feature_names=["fea_1", "fea_2", "fea_3"])
   
   # 拟合并转换数据
   new_features_data = cc.fit_transform(data)
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = cc.get_feature_names_out()
   print("new_features_list: ", list(new_features_list))
   ```

   运行结果
   ```txt
   new_features_data: 
      CrossCombination_fea_1_&_fea_3 CrossCombination_fea_1_&_fea_2  \
   0                          Q_&_1                          Q_&_A   
   1                          W_&_1                          W_&_B   
   2                          W_&_3                          W_&_A   
   3                          Q_&_4                          Q_&_C   
   
     CrossCombination_fea_2_&_fea_3  
   0                          A_&_1  
   1                          B_&_1  
   2                          A_&_3  
   3                          C_&_4  
   new_features_list:  ['CrossCombination_fea_1_&_fea_3', 'CrossCombination_fea_1_&_fea_2', 'CrossCombination_fea_2_&_fea_3']
   ```
   
2. 双变量交叉组合并对新特征执行 One-Hot 编码
   ```python
   import pandas as pd
   
   from alpha_feature_tools import derivation
   
   
   # 创建数据
   data = pd.DataFrame({
       "fea_1": ["Q", "W", "W", "Q"],
       "fea_2": ["A", "B", "A", "C"],
       "fea_3": [1, 1, 3, 4],
       "fea_4": [6, 6, 6, 7]
   })
   
   # 示例化 CrossCombination 转化器
   cc = derivation.CrossCombination(feature_names=["fea_1", "fea_2", "fea_3"], is_one_hot=True)
   
   # 拟合并转换数据
   new_features_data = cc.fit_transform(data)
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = cc.get_feature_names_out()
   print("new_features_list: ", list(new_features_list))
   ```

   运行结果
   ```txt
   new_features_data: 
       CrossCombination_fea_1_&_fea_3_Q_&_1  CrossCombination_fea_1_&_fea_3_Q_&_4  \
   0                                   1.0                                   0.0   
   1                                   0.0                                   0.0   
   2                                   0.0                                   0.0   
   3                                   0.0                                   1.0   
   
      CrossCombination_fea_1_&_fea_3_W_&_1  CrossCombination_fea_1_&_fea_3_W_&_3  \
   0                                   0.0                                   0.0   
   1                                   1.0                                   0.0   
   2                                   0.0                                   1.0   
   3                                   0.0                                   0.0   
   
      CrossCombination_fea_1_&_fea_2_Q_&_A  CrossCombination_fea_1_&_fea_2_Q_&_C  \
   0                                   1.0                                   0.0   
   1                                   0.0                                   0.0   
   2                                   0.0                                   0.0   
   3                                   0.0                                   1.0   
   
      CrossCombination_fea_1_&_fea_2_W_&_A  CrossCombination_fea_1_&_fea_2_W_&_B  \
   0                                   0.0                                   0.0   
   1                                   0.0                                   1.0   
   2                                   1.0                                   0.0   
   3                                   0.0                                   0.0   
   
      CrossCombination_fea_2_&_fea_3_A_&_1  CrossCombination_fea_2_&_fea_3_A_&_3  \
   0                                   1.0                                   0.0   
   1                                   0.0                                   0.0   
   2                                   0.0                                   1.0   
   3                                   0.0                                   0.0   
   
      CrossCombination_fea_2_&_fea_3_B_&_1  CrossCombination_fea_2_&_fea_3_C_&_4  
   0                                   0.0                                   0.0  
   1                                   1.0                                   0.0  
   2                                   0.0                                   0.0  
   3                                   0.0                                   1.0  
   new_features_list:  ['CrossCombination_fea_1_&_fea_3_Q_&_1', 'CrossCombination_fea_1_&_fea_3_Q_&_4', 'CrossCombination_fea_1_&_fea_3_W_&_1', 'CrossCombination_fea_1_&_fea_3_W_&_3', 'CrossCombination_fea_1_&_fea_2_Q_&_A', 'CrossCombination_fea_1_&_fea_2_Q_&_C', 'CrossCombination_fea_1_&_fea_2_W_&_A', 'CrossCombination_fea_1_&_fea_2_W_&_B', 'CrossCombination_fea_2_&_fea_3_A_&_1', 'CrossCombination_fea_2_&_fea_3_A_&_3', 'CrossCombination_fea_2_&_fea_3_B_&_1', 'CrossCombination_fea_2_&_fea_3_C_&_4']
   ```
   
3. 三变量交叉组合
   ```python
   import pandas as pd
   
   from alpha_feature_tools import derivation
   
   
   # 创建数据
   data = pd.DataFrame({
       "fea_1": ["Q", "W", "W", "Q"],
       "fea_2": ["A", "B", "A", "C"],
       "fea_3": [1, 1, 3, 4],
       "fea_4": [6, 6, 6, 7]
   })
   
   # 示例化 CrossCombination 转化器
   cc = derivation.CrossCombination(feature_names=["fea_1", "fea_2", "fea_3", "fea_4"], n=3)
   
   # 拟合并转换数据
   new_features_data = cc.fit_transform(data)
   print("new_features_data: \n", new_features_data)
   
   # 获取衍生后的新特征列表
   new_features_list = cc.get_feature_names_out()
   print("new_features_list: ", list(new_features_list))
   ```

   运行结果
   ```txt
   new_features_data: 
      CrossCombination_fea_1_&_fea_2_&_fea_4  \
   0                              Q_&_A_&_6   
   1                              W_&_B_&_6   
   2                              W_&_A_&_6   
   3                              Q_&_C_&_7   
   
     CrossCombination_fea_2_&_fea_3_&_fea_4  \
   0                              A_&_1_&_6   
   1                              B_&_1_&_6   
   2                              A_&_3_&_6   
   3                              C_&_4_&_7   
   
     CrossCombination_fea_1_&_fea_3_&_fea_4  \
   0                              Q_&_1_&_6   
   1                              W_&_1_&_6   
   2                              W_&_3_&_6   
   3                              Q_&_4_&_7   
   
     CrossCombination_fea_1_&_fea_2_&_fea_3  
   0                              Q_&_A_&_1  
   1                              W_&_B_&_1  
   2                              W_&_A_&_3  
   3                              Q_&_C_&_4  
   new_features_list:  ['CrossCombination_fea_1_&_fea_2_&_fea_4', 'CrossCombination_fea_2_&_fea_3_&_fea_4', 'CrossCombination_fea_1_&_fea_3_&_fea_4', 'CrossCombination_fea_1_&_fea_2_&_fea_3']
   ```

