from sklearn.feature_selection import RFE as SKL_RFE
from sklearn.feature_selection import RFECV as SKL_RFECV


class RFE(SKL_RFE):
    """Feature ranking with recursive feature elimination.

       Given an external estimator that assigns weights to features (e.g., the
       coefficients of a linear model), the goal of recursive feature elimination
       (RFE) is to select features by recursively considering smaller and smaller
       sets of features. First, the estimator is trained on the initial set of
       features and the importance of each feature is obtained either through
       any specific attribute or callable.
       Then, the least important features are pruned from current set of features.
       That procedure is recursively repeated on the pruned set until the desired
       number of features to select is eventually reached.

       Read more in the :ref:`User Guide <rfe>`.

       Parameters
       ----------
       estimator : ``Estimator`` instance
           A supervised learning estimator with a ``fit`` method that provides
           information about feature importance
           (e.g. `coef_`, `feature_importances_`).

       n_features_to_select : int or float, default=None
           The number of features to select. If `None`, half of the features are
           selected. If integer, the parameter is the absolute number of features
           to select. If float between 0 and 1, it is the fraction of features to
           select.

       step : int or float, default=1
           If greater than or equal to 1, then ``step`` corresponds to the
           (integer) number of features to remove at each iteration.
           If within (0.0, 1.0), then ``step`` corresponds to the percentage
           (rounded down) of features to remove at each iteration.

       verbose : int, default=0
           Controls verbosity of output.

       importance_getter : str or callable, default='auto'
           If 'auto', uses the feature importance either through a `coef_`
           or `feature_importances_` attributes of estimator.

           Also accepts a string that specifies an attribute name/path
           for extracting feature importance (implemented with `attrgetter`).
           For example, give `regressor_.coef_` in case of
           :class:`~sklearn.compose.TransformedTargetRegressor`  or
           `named_steps.clf.feature_importances_` in case of
           class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

           If `callable`, overrides the default feature importance getter.
           The callable is passed with the fitted estimator and it should
           return importance for each feature.

       Examples
       --------
       The following example shows how to retrieve the 5 most informative
       features in the Friedman #1 dataset.

       >>> from sklearn.datasets import make_friedman1
       >>> from sklearn.feature_selection import RFE
       >>> from sklearn.svm import SVR
       >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
       >>> estimator = SVR(kernel="linear")
       >>> selector = RFE(estimator, n_features_to_select=5, step=1)
       >>> selector = selector.fit(X, y)
       >>> selector.support_
       array([ True,  True,  True,  True,  True, False, False, False, False,
              False])
       >>> selector.ranking_
       array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
       """
    pass


class RFECV(SKL_RFECV):
    """Recursive feature elimination with cross-validation to select features.

      See glossary entry for :term:`cross-validation estimator`.

      Read more in the :ref:`User Guide <rfe>`.

      Parameters
      ----------
      estimator : ``Estimator`` instance
          A supervised learning estimator with a ``fit`` method that provides
          information about feature importance either through a ``coef_``
          attribute or through a ``feature_importances_`` attribute.

      step : int or float, default=1
          If greater than or equal to 1, then ``step`` corresponds to the
          (integer) number of features to remove at each iteration.
          If within (0.0, 1.0), then ``step`` corresponds to the percentage
          (rounded down) of features to remove at each iteration.
          Note that the last iteration may remove fewer than ``step`` features in
          order to reach ``min_features_to_select``.

      min_features_to_select : int, default=1
          The minimum number of features to be selected. This number of features
          will always be scored, even if the difference between the original
          feature count and ``min_features_to_select`` isn't divisible by
          ``step``.

      cv : int, cross-validation generator or an iterable, default=None
          Determines the cross-validation splitting strategy.
          Possible inputs for cv are:

          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

          For integer/None inputs, if ``y`` is binary or multiclass,
          :class:`~sklearn.model_selection.StratifiedKFold` is used. If the
          estimator is a classifier or if ``y`` is neither binary nor multiclass,
          :class:`~sklearn.model_selection.KFold` is used.

          Refer :ref:`User Guide <cross_validation>` for the various
          cross-validation strategies that can be used here.

      scoring : str, callable or None, default=None
          A string (see model evaluation documentation) or
          a scorer callable object / function with signature
          ``scorer(estimator, X, y)``.

      verbose : int, default=0
          Controls verbosity of output.

      n_jobs : int or None, default=None
          Number of cores to run in parallel while fitting across folds.
          ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
          ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
          for more details.

      importance_getter : str or callable, default='auto'
          If 'auto', uses the feature importance either through a `coef_`
          or `feature_importances_` attributes of estimator.

          Also accepts a string that specifies an attribute name/path
          for extracting feature importance.
          For example, give `regressor_.coef_` in case of
          :class:`~sklearn.compose.TransformedTargetRegressor`  or
          `named_steps.clf.feature_importances_` in case of
          :class:`~sklearn.pipeline.Pipeline` with its last step named `clf`.

          If `callable`, overrides the default feature importance getter.
          The callable is passed with the fitted estimator and it should
          return importance for each feature.

      Examples
      --------
      The following example shows how to retrieve the a-priori not known 5
      informative features in the Friedman #1 dataset.

      >>> from sklearn.datasets import make_friedman1
      >>> from sklearn.feature_selection import RFECV
      >>> from sklearn.svm import SVR
      >>> X, y = make_friedman1(n_samples=50, n_features=10, random_state=0)
      >>> estimator = SVR(kernel="linear")
      >>> selector = RFECV(estimator, step=1, cv=5)
      >>> selector = selector.fit(X, y)
      >>> selector.support_
      array([ True,  True,  True,  True,  True, False, False, False, False,
             False])
      >>> selector.ranking_
      array([1, 1, 1, 1, 1, 6, 4, 3, 2, 5])
      """
    pass
