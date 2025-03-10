# NLP 转化器




## 一、简介

NLP 转化器是用于在给定的数据集中将多个 **离散型** 特征通过词向量衍生出新的特征。

该转化器是一个自定义的 Transformer 类，继承于 sklearn.base 中的 TransformerMixin 和 BaseEstimator，因此在该类中实现了 fit() 和 transform() 方法，并在调用时可以使用父类中的 fit_transform() 方法，同时该转换器也可以无缝集成到 scikit-learn 的流水线中。



## 二、参数说明

| 参数名         | 类型   | 默认值  | 描述               |
| :------------- |:-----|:----:|:-----------------|
| is_tf_idf | bool | True | 是否需要进行 TF-IDF 编码 |



## 三、使用方法

加载数据后先使用 fit() 方法进行拟合，然后使用 transform() 方法进行转换，或者直接使用 fit_transform() 方法进行拟合转换即可。



## 四、示例

```python
import pandas as pd

from alpha_feature_tools.derivation import NLP

X = pd.DataFrame(
    data=[
        [0, 1, 0, 2, 3],
        [3, 2, 1, 0, 0]
    ],
    columns=["a", "b", "c", "d", "e"]
)

nlp = NLP(is_tf_idf=False)
nlp.fit(X)
out = nlp.transform(X)
print(nlp.get_feature_names_out())
print(out)

nlp = NLP(is_tf_idf=True)
nlp.fit(X)
out = nlp.transform(X)
print(nlp.get_feature_names_out())
print(out)
```

运行结果
```txt
['NLP_a', 'NLP_b', 'NLP_c', 'NLP_d', 'NLP_e']
[[0 1 0 1 1]
 [1 1 1 0 0]]
['NLP_a_tf_idf', 'NLP_b_tf_idf', 'NLP_c_tf_idf', 'NLP_d_tf_idf', 'NLP_e_tf_idf']
[[0.         2.09861229 0.         2.79175947 2.79175947]
 [2.79175947 2.09861229 2.79175947 0.         0.        ]]
```
