from sklearn.preprocessing import StandardScaler as SKL_StandardScaler
from sklearn.preprocessing import MinMaxScaler as SKL_MinMaxScaler
from sklearn.preprocessing import Normalizer as SKL_Normalizer


class StandardScaler(SKL_StandardScaler):
    """Standardize features by removing the mean and scaling to unit variance.

    The standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples or zero if `with_mean=False`,
    and `s` is the standard deviation of the training samples or one if
    `with_std=False`.

    Centering and scaling happen independently on each feature by computing
    the relevant statistics on the samples in the training set. Mean and
    standard deviation are then stored to be used on later data using
    :meth:`transform`.

    Standardization of a dataset is a common requirement for many
    machine learning estimators: they might behave badly if the
    individual features do not more or less look like standard normally
    distributed data (e.g. Gaussian with 0 mean and unit variance).

    For instance many elements used in the objective function of
    a learning algorithm (such as the RBF kernel of Support Vector
    Machines or the L1 and L2 regularizers of linear models) assume that
    all features are centered around 0 and have variance in the same
    order. If a feature has a variance that is orders of magnitude larger
    than others, it might dominate the objective function and make the
    estimator unable to learn from other features correctly as expected.

    This scaler can also be applied to sparse CSR or CSC matrices by passing
    `with_mean=False` to avoid breaking the sparsity structure of the data.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    copy : bool, default=True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

    with_mean : bool, default=True
        If True, center the data before scaling.
        This does not work (and will raise an exception) when attempted on
        sparse matrices, because centering them entails building a dense
        matrix which in common use cases is likely to be too large to fit in
        memory.

    with_std : bool, default=True
        If True, scale the data to unit variance (or equivalently,
        unit standard deviation).

    Attributes
    ----------
    scale_ : ndarray of shape (n_features,) or None
        Per feature relative scaling of the data to achieve zero mean and unit
        variance. Generally this is calculated using `np.sqrt(var_)`. If a
        variance is zero, we can't achieve unit variance, and the data is left
        as-is, giving a scaling factor of 1. `scale_` is equal to `None`
        when `with_std=False`.

    mean_ : ndarray of shape (n_features,) or None
        The mean value for each feature in the training set.
        Equal to ``None`` when ``with_mean=False``.

    var_ : ndarray of shape (n_features,) or None
        The variance for each feature in the training set. Used to compute
        `scale_`. Equal to ``None`` when ``with_std=False``.

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    n_samples_seen_ : int or ndarray of shape (n_features,)
        The number of samples processed by the estimator for each feature.
        If there are no missing samples, the ``n_samples_seen`` will be an
        integer, otherwise it will be an array of dtype int. If
        `sample_weights` are used it will be a float (if no missing data)
        or an array of dtype float that sums the weights seen so far.
        Will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    Examples
    --------
    >>> from alpha_feature_tools.preprocess import StandardScaler
    >>> data = [[0, 0], [0, 0], [1, 1], [1, 1]]
    >>> scaler = StandardScaler()
    >>> print(scaler.fit(data))
    StandardScaler()
    >>> print(scaler.mean_)
    [0.5 0.5]
    >>> print(scaler.transform(data))
    [[-1. -1.]
     [-1. -1.]
     [ 1.  1.]
     [ 1.  1.]]
    >>> print(scaler.transform([[2, 2]]))
    [[3. 3.]]
    """
    pass


class MinMaxScaler(SKL_MinMaxScaler):
    """Transform features by scaling each feature to a given range.

    This estimator scales and translates each feature individually such
    that it is in the given range on the training set, e.g. between
    zero and one.

    The transformation is given by::

        X_std = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
        X_scaled = X_std * (max - min) + min

    where min, max = feature_range.

    This transformation is often used as an alternative to zero mean,
    unit variance scaling.

    Read more in the :ref:`User Guide <preprocessing_scaler>`.

    Parameters
    ----------
    feature_range : tuple (min, max), default=(0, 1)
        Desired range of transformed data.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array).

    clip : bool, default=False
        Set to True to clip transformed values of held-out data to
        provided `feature range`.

    Attributes
    ----------
    min_ : ndarray of shape (n_features,)
        Per feature adjustment for minimum. Equivalent to
        ``min - X.min(axis=0) * self.scale_``

    scale_ : ndarray of shape (n_features,)
        Per feature relative scaling of the data. Equivalent to
        ``(max - min) / (X.max(axis=0) - X.min(axis=0))``

    data_min_ : ndarray of shape (n_features,)
        Per feature minimum seen in the data

    data_max_ : ndarray of shape (n_features,)
        Per feature maximum seen in the data

    data_range_ : ndarray of shape (n_features,)
        Per feature range ``(data_max_ - data_min_)`` seen in the data

    n_features_in_ : int
        Number of features seen during :term:`fit`.

    n_samples_seen_ : int
        The number of samples processed by the estimator.
        It will be reset on new calls to fit, but increments across
        ``partial_fit`` calls.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.preprocessing import MinMaxScaler
    >>> data = [[-1, 2], [-0.5, 6], [0, 10], [1, 18]]
    >>> scaler = MinMaxScaler()
    >>> print(scaler.fit(data))
    MinMaxScaler()
    >>> print(scaler.data_max_)
    [ 1. 18.]
    >>> print(scaler.transform(data))
    [[0.   0.  ]
     [0.25 0.25]
     [0.5  0.5 ]
     [1.   1.  ]]
    >>> print(scaler.transform([[2, 2]]))
    [[1.5 0. ]]
    """
    pass


class Normalizer(SKL_Normalizer):
    """Normalize samples individually to unit norm.

    Each sample (i.e. each row of the data matrix) with at least one
    non zero component is rescaled independently of other samples so
    that its norm (l1, l2 or inf) equals one.

    This transformer is able to work both with dense numpy arrays and
    scipy.sparse matrix (use CSR format if you want to avoid the burden of
    a copy / conversion).

    Scaling inputs to unit norms is a common operation for text
    classification or clustering for instance. For instance the dot
    product of two l2-normalized TF-IDF vectors is the cosine similarity
    of the vectors and is the base similarity metric for the Vector
    Space Model commonly used by the Information Retrieval community.

    Read more in the :ref:`User Guide <preprocessing_normalization>`.

    Parameters
    ----------
    norm : {'l1', 'l2', 'max'}, default='l2'
        The norm to use to normalize each non zero sample. If norm='max'
        is used, values will be rescaled by the maximum of the absolute
        values.

    copy : bool, default=True
        Set to False to perform inplace row normalization and avoid a
        copy (if the input is already a numpy array or a scipy.sparse
        CSR matrix).

    Attributes
    ----------
    n_features_in_ : int
        Number of features seen during :term:`fit`.

    feature_names_in_ : ndarray of shape (`n_features_in_`,)
        Names of features seen during :term:`fit`. Defined only when `X`
        has feature names that are all strings.

    Examples
    --------
    >>> from sklearn.preprocessing import Normalizer
    >>> X = [[4, 1, 2, 2],
    ...      [1, 3, 9, 3],
    ...      [5, 7, 5, 1]]
    >>> transformer = Normalizer().fit(X)  # fit does nothing.
    >>> transformer
    Normalizer()
    >>> transformer.transform(X)
    array([[0.8, 0.2, 0.4, 0.4],
           [0.1, 0.3, 0.9, 0.3],
           [0.5, 0.7, 0.5, 0.1]])
    """
    pass
