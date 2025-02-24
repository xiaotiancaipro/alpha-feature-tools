# StandardScaler 转化器




## 一、简介

StandardScaler 转化器用于对数据进行标准化，将每个特征缩放到均值为0，标准差为1，使特征尺度一致。其原理是基于统计学中的 z-score 方法，消除特征间尺度差异，优化模型性能，适用于不同尺度特征或数据分布不均衡的情况。

该转化器继承自 sklearn.preprocessing.StandardScaler。



## 二、参数说明

| 参数名            | 类型      | 默认值   | 描述                                               |
|:----------------|:---------|:-------:|:-------------------------------------------------- |
| copy            | bool     | True   | 如果为False，尝试避免复制并进行原地缩放。这不能保证总是原地进行；例如，如果数据不是NumPy数组或scipy.sparse CSR矩阵，仍可能返回一个副本。 |
| with_mean       | bool     | True   | 如果为True，在缩放前中心化数据。这在尝试对稀疏矩阵执行时会导致错误，因为对其进行中心化会生成一个密集矩阵，在常用情况下可能会太大而无法容纳在内存中。 |
| with_std        | bool     | True   | 如果为True，在缩放后将数据标准化为单位方差（等价于单位标准差）。 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
import pandas as pd

from alpha_feature_tools.preprocess import StandardScaler

data = pd.DataFrame(data=[[1, 223, 0], [2, 653, 0], [9, 100, 1]], columns=["a", "b", "label"])
X = data[["a", "b"]]
y = data["label"]

ss = StandardScaler()
ss.fit(X, y)
out = ss.transform(X)
print(ss.get_feature_names_out())
print(out)
```

运行结果
```txt
['a' 'b']
[[-0.84292723 -0.4316509 ]
 [-0.56195149  1.38212649]
 [ 1.40487872 -0.95047559]]
```

