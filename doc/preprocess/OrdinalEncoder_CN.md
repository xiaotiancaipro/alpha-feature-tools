# OrdinalEncoder 转化器




## 一、简介

OrdinalEncoder 转化器通过将有序类别编码为离散整数，保留顺序信息，优化了特征表示。适用于评分、排名等任务，帮助模型更好地捕捉数据关系。合理配置参数，确保在数据预处理和模型训练中有效支持任务需求。

该转化器继承自 sklearn.preprocessing.OrdinalEncoder。



## 二、参数说明

| 参数名            | 类型                     |  默认值   | 描述                                               |
|:---------------|:-----------------------|:------:| :------------------------------------------------- |
| categories     | {'auto', list}         | 'auto' | 自动确定类别或从列表中的每个元素对应特征的预期类别 |
| dtype          | num                    | numpy.float64 | 变量转换后的数据类型 |
| handle_unknown | str                    | 'error' | 处理未知类别的方式，'error'或'use_encoded_value' |
| unknown_value  | {int, numpy.nan, None} | None | 未知类别的编码值，当 handle_unknown 为 'use_encoded_value' 时，该参数才有效 |
| encoded_missing_value  | {int, np.nan}                     | np.nan | 缺失类别的编码值 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
import pandas as pd

from alpha_feature_tools.preprocess import OrdinalEncoder

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

oe = OrdinalEncoder()
oe.fit(X, y)
print(oe.transform(X))
print(oe.get_feature_names_out())
```

运行结果
```txt
[[1. 0.]
 [0. 0.]
 [0. 1.]]
['sex' 'city']
```

