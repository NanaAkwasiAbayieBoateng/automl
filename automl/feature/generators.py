"""Pipeline steps for feature generation"""

import logging
import itertools as it
from functools import partial
import random
import numpy as np
from sklearn.preprocessing import PolynomialFeatures


class PolynomialFeatureGenerator:
    """Generate polynomial and interaction features for dataset in PipelineData

    Generate a new feature matrix from current matrix in pipeline
    consisting of all polynomial combinations of the features with
    degree less than or equal to the specified degree. For example,
    if an input sample is two dimensional and of the form [a, b],
    the degree-2 polynomial features are [1, a, b, a^2, ab, b^2].

    Parameters
    ----------
    max_degree : integer
        The maximum degree of the polynomial features. Default = 2.

    Notes
    -----
    Be aware that the number of features in the output array scales
    polynomially in the number of features of the input array, and
    exponentially in the degree. High degrees can cause overfitting.
    """
    def __init__(self, max_degree):
        self._log = logging.getLogger(self.__class__.__name__)
        self.max_degree = max_degree

    def __call__(self, pipeline_data, pipeline_context):
        """
        Parametrs
        ---------
        pipeline_data : PipelineData
            Data passed between PipelineStep in pipeline

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        PipelineData
            PipelineData containing changed PipelineData.dataset
        """
        data = pipeline_data.dataset.data
        meta = pipeline_data.dataset.meta
        orig_feature_num = pipeline_data.dataset.data.shape[1]

        for degree in range(1, self.max_degree+1):
            # TODO this is unreadable, unwrap
            sets_of_indices = list(set(tuple(sorted(indices)) for indices in it.product(range(0, orig_feature_num), repeat=degree)))

            for indices in sets_of_indices:
                new_feature = np.ones((data.shape[0], 1), dtype='float32')
                history = ""

                for index in indices:
                    new_feature = np.reshape(
                        data[:, index],
                        (data.shape[0], 1)
                    ) * new_feature
                    history = meta[index]['history'] + '*' + history
                if np.isfinite(new_feature).all():
                    data = np.append(data, new_feature, axis=1)
                    meta.append({
                        "name": "",
                        "history": history[:-1]  # drop last symbol
                    })
                else:
                    pass

        pipeline_data.dataset.data = data
        pipeline_data.dataset.meta = meta
        return pipeline_data


class SklearnFeatureGenerator:
    """ Wrapper for Scikit-Learn Transformers (only for
    sklearn.preprocessing.PolynomialFeatures at this release)

    Parameters
    ----------
    transformer_class: sklearn.preprocessing.PolynomialFeatures

    args:
        arguments are passed to sklearn PolynomialFeatures

    kwargs:
        keyword arguments are passed to sklearn PolynomialFeatures
    """
    def __init__(self, transformer_class, *args, **kwargs):
        self._log = logging.getLogger(self.__class__.__name__)
        self._transformer = transformer_class(*args, **kwargs)

    def __call__(self, pipeline_data, pipeline_context):
        """
        Parameters
        ----------
        pipeline_data : PipelineData
            Data passed between PipelineStep in pipeline

        pipeline_context: automl.pipeline.PipelineContext
            global context of a pipeline

        Returns
        -------
        PipelineData
            PipelineData containing changed PipelineData.dataset
        """

        pipeline_data.dataset.data = self._transformer.fit_transform(pipeline_data.dataset.data)
        return pipeline_data


class FormulaFeatureGenerator:
    """ Generate features for dataset in PipelineData by formula

    Generate a new feature matrix from current matrix in pipeline
    consisting of old features and new features generated from operations
    over old features

    Paramets
    --------
    func_list : list of symbols of functions
        In current version func_list may contain only '+', '-', '*', '/'.

    limit: int
        Total number of expression members

    Attributes
    ----------
    _func_map : dict of generating functions
        The function for generation new features

    used_func : set of of symbols of functions
    """

    def __init__(self, func_list=['+', '-', '*', '/'], limit=1):
        self._log = logging.getLogger(self.__class__.__name__)
        self.used_func = set(func_list)
        self._func_map = {
            '+': self._sum,
            '-': self._substract,
            '/': self._divide,
            '*': self._multiply,
        }
        self._limit = limit

    def _sum(self, dataset):
        """ Generate one new feature by sum of two random features

        Parametrs
        ---------
        dataset : Dataset
            Dataset.data contains data to transform

        Returns
        -------
        np.ndarray shape [n_sample, 1]
            new feature

        history : str
            the history of new feature for reproducible preprocessing
        """
        X = dataset.data
        first_index, second_index = self._choose_two_index(X)
        x, y = X[:, first_index].reshape(
            X.shape[0], 1), X[:, second_index].reshape(X.shape[0], 1)
        history = f"({dataset.meta[first_index]['history']}+{dataset.meta[second_index]['history']})"
        name = f"{dataset.meta[first_index]['name']}_+_{dataset.meta[second_index]['name']}"
        return x + y, history, name

    def _substract(self, dataset):
        """Generate one new feature by substraction of two random features

        Parametrs
        ---------
        dataset : Dataset
            Dataset.data contains data to transform

        Returns
        -------
        np.ndarray shape [n_sample, 1]
            new feature

        history : str
            the history of new feature for reproducible preprocessing
        """
        X = dataset.data
        first_index, second_index = self._choose_two_index(X)
        x, y = X[:, first_index].reshape(
            X.shape[0], 1), X[:, second_index].reshape(X.shape[0], 1)
        history = f"({dataset.meta[first_index]['history']}-{dataset.meta[second_index]['history']})"
        name = f"{dataset.meta[first_index]['name']}_-_{dataset.meta[second_index]['name']}"
        return x - y, history, name

    def _divide(self, dataset):
        """ Generate one new feature by division of two random features

        Parametrs
        ---------
        dataset : Dataset
            Dataset.data contains data to transform
            
        Returns
        -------
        np.ndarray shape [n_sample, 1]
            new feature

        history : str
            the history of new feature for reproducible preprocessing
        """
        X = dataset.data
        first_index, second_index = self._choose_two_index(X)
        x, y = X[:, first_index].reshape(X.shape[0], 1), X[:, second_index].reshape(X.shape[0], 1)
        history = f"({dataset.meta[first_index]['history']}/{dataset.meta[second_index]['history']})"
        name = f"{dataset.meta[first_index]['name']}_/_{dataset.meta[second_index]['name']}"
        return x / y, history, name

    def _multiply(self, dataset):
        """ Generate one new feature by multiplication of two random features

        Parametrs
        ---------
        dataset : Dataset
            Dataset.data contains data to transform

        Returns
        -------
        np.ndarray shape [n_sample, 1]
            new feature

        history : str
            the history of new feature for reproducible preprocessing
        """
        X = dataset.data
        first_index, second_index = self._choose_two_index(X)
        x, y = X[:, first_index].reshape(X.shape[0], 1), X[:, second_index].reshape(X.shape[0], 1)
        history = f"({dataset.meta[first_index]['history']}*{dataset.meta[second_index]['history']})"
        name = f"{dataset.meta[first_index]['name']}_*_{dataset.meta[second_index]['name']}"
        return x * y, history, name

    def _choose_two_index(self, X):
        """ Choose two index input data

        Parametrs
        ---------
        X : np.ndarray, shape [n_samples, n_features]
            The data

        Returns
        -------
        int: 
            Index

        int:
            Index 
        """
        return random.randint(0, X.shape[1]-1),\
               random.randint(0, X.shape[1]-1)

    def __call__(self, pipeline_data, pipeline_context):
        """
        Parameters
        ----------
        pipeline_data : automl.pipeline.PipelineData
            Data passed between PipelineStep in pipeline

        pipeline_context : automl.pipeline.PipelineContext
            global context of a pipeline

        limit : int
            Maximum amount of new features. Default = 10

        Returns
        -------
        PipelineData
            PipelineData containing changed PipelineData.dataset
        """
        orig_feature_num = pipeline_data.dataset.data.shape[1]

        for _ in range(0, self._limit):
            new_feature, history, name = self._func_map[random.sample(self.used_func, 1)[0]](pipeline_data.dataset)
            if np.isfinite(new_feature).all():
                pipeline_data.dataset.data = np.append(pipeline_data.dataset.data, new_feature, axis=1)

                pipeline_data.dataset.meta.append({
                    "name": name,
                    "history": history
                })

        self._log.info((f"Generated new features. Old feature number - "
                        f"{orig_feature_num}, new feature number - "
                        f"{pipeline_data.dataset.data.shape[1]}"))
        return pipeline_data 

class Preprocessing:
    """ Reproduce all feature transformations that were creating during the
    execution of AutoML pipeline
    """

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)

    def reproduce(self, resulting_dataset, original_dataset):
        """ Parameters
        ----------
        resulting_dataset : Dataset
            Resulting dataset

        original_dataset : Dataset
            Initial dataset that was passed to executor
            
        Returns
        -------
        np.ndarray
            Dataset that was generated by executor
        """
        data = original_dataset.data.astype("float32")
        final_data = np.ones((data.shape[0], 1), dtype='float32')
        for feature in resulting_dataset.meta:
            explicit_locals = locals()
            exec(
                f"new_feature = {feature['history']}",
                globals(),
                explicit_locals
            )
            new_feature = np.reshape(
                explicit_locals["new_feature"],
                (data.shape[0], 1)
            )
            final_data = np.append(final_data, new_feature, axis=1) 
        return np.delete(final_data.astype('float32'), 0, 1)

PolynomialGenerator = partial(SklearnFeatureGenerator, PolynomialFeatures)
