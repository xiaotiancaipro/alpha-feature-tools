# Normalizer 转化器




## 一、简介

Normalizer 转化器用于将样本独立归一化为单位范数。 每个样本（即数据矩阵的每一行）至少有一个非零分量，会独立于其他样本重新缩放，使其范数（l1、l2或无穷大）等于 1。

该转化器继承自 sklearn.preprocessing.Normalizer。



## 二、参数说明

| 参数名            | 类型      | 默认值   | 描述                                               |
|:----------------|:---------|:--------:|:-------------------------------------------------- |
| norm            | {'l1', 'l2', 'max'} | 'l2'   | 使用来归范每个非零样本的范数。如果使用 norm='max'，值将被最大绝对值进行缩放。 |
| copy            | bool     | True    | 设置为 False 可以进行原地行归范并且避免复制（如果输入已经是 numpy 数组或 scipy 稀疏 CSR 矩阵）。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
from alpha_feature_tools.preprocess import Normalizer

X = [[4, 1, 2, 2], [1, 3, 9, 3], [5, 7, 5, 1]]
transformer = Normalizer()
transformer.fit(X)
out = transformer.transform(X)
print(out)
```

运行结果
```txt
[[0.8 0.2 0.4 0.4]
 [0.1 0.3 0.9 0.3]
 [0.5 0.7 0.5 0.1]]
```

