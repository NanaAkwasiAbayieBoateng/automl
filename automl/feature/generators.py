"""Feature Generators"""

import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import PolynomialFeatures
from functools import partial


class SklearnFeatureGenerator:
    def __init__(self, transformer_class, *args, **kwargs):
        """
        Wrapper for Scikit-Learn Transformers

        Parameters
        ----------
        kwargs:
            keyword arguments are passed to sklearn PolynomialFeatures
        """
        self._log = logging.getLogger(self.__class__.__name__)
        self._transformer = transformer_class(*args, **kwargs)

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
        """
        return self._transformer.fit_transform(X)


class FormulaFeatureGenerator:
    """TODO Doc"""
    def __init__(self, func_list=['+', '-']):
        raise NotImplementedError()


PolynomialGenerator = partial(SklearnFeatureGenerator, PolynomialFeatures)
