# OneHotEncoder 转化器




## 一、简介

OneHotEncoder 转化器将分类数据（离散和有序）转换为数值形式，通过创建一个新的二进制列来表示原始数据中的类别。

该转化器继承自 sklearn.preprocessing.OneHotEncoder。



## 二、参数说明

| 参数名            | 类型      | 默认值   | 描述                                               |
|:----------------|:---------|:-------:|:-------------------------------------------------- |
| **categories**   | {'auto', list} | 'auto' | 指定每个特征的类别。自动从训练数据中确定类别或指定类别列表。 |
| **drop**         | {'first', 'if_binary', array} | None   | 指定删除每个特征的一种类别，以解决类别完全棱角化的问题。 |
| **sparse**       | bool     | True   | 输出格式化为稀疏矩阵（True）还是密集数组（False）。 |
| **dtype**        | numeric type | float  | 输出数据类型，用于处理缺失值时需为float。 |
| **handle_unknown** | {'error', 'ignore', 'infrequent_if_exist'} | 'error' | 处理未知类别的方式。 |
| **min_frequency** | int or float | None   | 确定类别频率以识别稀疏类别，如int按卡Holder度数，float按比例计算。 |
| **max_categories** | int | None   | 每个特征输出的最大特征数，包括稀疏类别。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
import pandas as pd

from alpha_feature_tools.preprocess import OneHotEncoder

data = pd.DataFrame(
    data=[
        ['Male', "paris", 0],
        ['Female', "paris", 1],
        ['Female', "tokyo", 1]
    ],
    columns=["sex", "city", "label"]
)
X = data[["sex", "city"]]
y = data["label"]

oh = OneHotEncoder(sparse=False)
oh.fit(X, y)
print(oh.transform(X))
print(oh.get_feature_names_out())
```

运行结果
```txt
[[0. 1. 1. 0.]
 [1. 0. 1. 0.]
 [1. 0. 0. 1.]]
['sex_Female' 'sex_Male' 'city_paris' 'city_tokyo']
```

