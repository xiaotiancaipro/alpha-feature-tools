# BinaryCrossCombination 转化器




## 一、简介

BinaryCrossCombination 转化器是用于在给定的数据集中生成二元交叉组合特征。它能够将两个或多个**类别型特征**通过两两交叉组合衍生出一个或多个新特征，并可选择对这些新特征进行 One-Hot 编码。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名        | 类型      | 默认值 | 描述                                |
| :------------ | :-------- | :----: | :---------------------------------- |
| feature_names | List[str] |  必填  | 需要参与交叉组合的原始特征名称列表  |
| is_one_hot    | bool      |  True  | 是否对生成的新特征进行 One-Hot 编码 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

代码

```python
import pandas as pd

from alpha_feature_tools import derivation


# 创建数据
data = pd.DataFrame({
    "sex": ["M", "F", "M", "F"],
    "city": ["A", "B", "A", "C"],
    "age": [25, 30, 35, 40]
})

# 示例化 BinaryCrossCombination 转化器
bcc = derivation.BinaryCrossCombination(feature_names=["sex", "city"], is_one_hot=True)

# 拟合并转换数据
new_features_data = bcc.fit_transform(data)
print("new_features_data: \n", new_features_data)

# 获取衍生后的新特征列表
new_features_list = bcc.get_feature_names_out()
print("new_features_list: ", list(new_features_list))
```

运行结果

```txt
new_features_data: 
    sex_&_city_F_&_B  sex_&_city_F_&_C  sex_&_city_M_&_A
0               0.0               0.0               1.0
1               1.0               0.0               0.0
2               0.0               0.0               1.0
3               0.0               1.0               0.0
new_features_list:  ['sex_&_city_F_&_B', 'sex_&_city_F_&_C', 'sex_&_city_M_&_A']
```

