"""Feature Generators"""

import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import PolynomialFeatures



class FeatureGeneratorBase(ABC):
    """Base class for all feature generators"""
    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_features(self):
        """All feature generators must implement this method.
        Feature generator should not reduce the number of features already
        present."""
        pass


class PolynomialFeatureGenerator(FeatureGeneratorBase):
    """Class for polynomial features generators"""
    def __init__(self):
        super().__init__()
    
    def generate_features(self, X, **kwargs):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.
        degree : integer
            The degree of the polynomial features. Default = 2.
        interaction_only : boolean, default = False
            If true, only interaction features are produced: features that are
            products of at most ``degree`` *distinct* input features (so not
            ``x[1] ** 2``, ``x[0] * x[2] ** 3``, etc.).
        include_bias : boolean
            If True (default), then include a bias column, the feature in which
            all polynomial powers are zero (i.e. a column of ones - acts as an
            intercept term in a linear model).
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        See Also
        --------
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        """
        return PolynomialFeatures(**kwargs).fit_transform(X)

    def __call__(self, X, **kwargs):
        self.generate_features(X, **kwargs)


class ArithmeticFeatureGenerator(FeatureGeneratorBase):
    """TODO Doc"""
    def __init__(self):
        raise NotImplementedError()
