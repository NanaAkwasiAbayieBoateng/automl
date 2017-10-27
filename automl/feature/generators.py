"""Feature Generators"""

import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import PolynomialFeatures



class FeatureGeneratorBase(ABC):
    """DEPRECATED, to be deleted
    
    Base class for all feature generators"""
    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    @abstractmethod
    def generate_features(self):
        """All feature generators must implement this method.
        Feature generator should not reduce the number of features already
        present."""
        pass


class PolynomialFeatureGenerator:
    def __init__(self, **kwargs):
        """
        Initialize Polynomial Feature Generator

        Parameters
        ----------
        kwargs:
            keyword arguments are passed to sklearn PolynomialFeatures

        See Also
        --------
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        """
        super().__init__()
        self._log = logging.getLogger(self.__class__.__name__)
        self._polynomial_features = PolynomialFeatures(**kwargs)

    def __call__(self, X, pipeline_context):
        """
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data.

        pipeline_context: automl.pipeline.PipelineContext
            global context of a pipeline
        
        Returns
        -------
        X_new : numpy array of shape [n_samples, n_features_new]
            Transformed array.

        See Also
        --------
        http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.PolynomialFeatures.html
        """
        return self._polynomial_features.fit_transform(X)


class FormulaFeatureGenerator(FeatureGeneratorBase):
    """TODO Doc"""
    def __init__(self, func_list=['+', '-']):
        raise NotImplementedError()
