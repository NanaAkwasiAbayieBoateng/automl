"""Feature Generators"""

import logging
from abc import ABC, abstractmethod
from sklearn.preprocessing import PolynomialFeatures
from functools import partial
import random
import numpy as np


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
    def __init__(self):
        """
        Initialize Formula Feature Generator

        Parameters
        ----------
        kwargs:
            keyword arguments are passed to sklearn PolynomialFeatures
        """
        self._func_map = {
            '+': self._sum,
            '-': self._substract,
            '/': self._divide,
            '*': self._multiply,
        }

    def _sum(self, X):
        x, y = self.choice_two_features(X)
        print('sum')
        return np.append(X, x+y, axis=1)
    def _substract(self, X):
        x, y = self.choice_two_features(X)
        print('sub')
        return np.append(X, x-y, axis=1)
    def _divide(self, X):
        x, y = self.choice_two_features(X)
        print('div')
        return np.append(X, x/y, axis=1)
    def _multiply(self, X):
        x, y = self.choice_two_features(X)
        print('mul')
        return np.append(X, x*y, axis=1)

    def choice_two_features(self, X):
        return X[:, random.randint(0, X.shape[1]-1)].reshape(X.shape[0], 1), \
               X[:, random.randint(0, X.shape[1]-1)].reshape(X.shape[0], 1)

    
    def __call__(self, X, limit, pipeline_context):
        if type(X) != np.ndarray:
            print('sadfjsajdfhaklsjdhflk')
            X = np.array(X)
        print(type(X))
        for i in range(0, limit):
            X = self._func_map[random.choice(list(self._func_map))](X)
        return X
       
        
PolynomialGenerator = partial(SklearnFeatureGenerator, PolynomialFeatures)
