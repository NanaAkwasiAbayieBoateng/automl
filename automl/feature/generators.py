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
        super().__init__(self)
    
    def generate_features(self, X, **kwarg):
        """
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
