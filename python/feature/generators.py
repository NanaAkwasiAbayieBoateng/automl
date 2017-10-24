"""Feature Generators"""

import logging
from abc import ABC, abstractmethod


class FeatureGeneratorBase(ABC):
    """Base class for all feature generators"""
    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_features(self, limit=None):
        """All feature generators must implement this method.
        Feature generator should not reduce the number of features already
        present.

        Parameters
        ----------
        limit: int
            maxinum number of features to generate"""
        pass


class PolynomialFeatureGenerator(FeatureGeneratorBase):
    """TODO Doc"""
    def __init__(self):
        raise NotImplementedError()


class ArithmeticFeatureGenerator(FeatureGeneratorBase):
    """TODO Doc"""
    def __init__(self):
        raise NotImplementedError()
