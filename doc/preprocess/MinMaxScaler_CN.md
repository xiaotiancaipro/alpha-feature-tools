# MinMaxScaler 转化器




## 一、简介

MinMaxScaler 转化器是一种用于标准化数据的方法，用于将数据缩放到一个特定的范围（通常是 0 到 1）。其主要目标是消除不同特征之间的尺度差异，使得数据具有更一致的尺度，便于模型训练和比较。

该转化器继承自 sklearn.preprocessing.MinMaxScaler。



## 二、参数说明

| 参数名            | 类型      | 默认值   | 描述                                               |
|:-----------------|:---------|:-------:|:-------------------------------------------------- |
| feature_range    | tuple    | (0, 1) | 转换后数据的期望范围                               |
| copy            | bool     | True   | 设置为 False 可以进行就地行标准化并避免复制（如果输入已经是 numpy 数组）。 |
| clip            | bool     | False  | 设置为 True 可以将保留数据中转换后的值限制在提供的 `feature range` 内。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
import pandas as pd

from alpha_feature_tools.preprocess import MinMaxScaler

data = pd.DataFrame(data=[[1, 223, 0], [2, 653, 0], [9, 100, 1]], columns=["a", "b", "label"])
X = data[["a", "b"]]
y = data["label"]

mms = MinMaxScaler()
mms.fit(X, y)
out = mms.transform(X)
print(mms.get_feature_names_out())
print(out)
```

运行结果
```txt
['a' 'b']
[[0.         0.22242315]
 [0.125      1.        ]
 [1.         0.        ]]
```

