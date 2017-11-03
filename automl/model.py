import multiprocessing
import operator
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split
from hyperopt import STATUS_OK

class ModelSpace:
    """
    Should include:
    Logistic Regression
    KNN classification
    Support Vector Machines
    RandomForest
    Gradient Boosting
    """
    def __init__(self, model_list):
        self._model_list = model_list

    def __call__(self, dataset, context):
        context.model_space = self._model_list
        return dataset

class CV:
    def __init__(self, n_folds=5, n_jobs=None, hyperopt_eval_format=False):
        self._n_folds = n_folds

        if n_jobs is None:
            self._n_jobs = multiprocessing.cpu_count()
        else:
            self._n_jobs = n_jobs

        self._hyperopt = hyperopt_eval_format

    def __call__(self, dataset, context, model_params=None):
        cv_results = [] 
       
        # TODO hyperopt won't work with multiple models
        # We should probabily leave CV evaluate single model and
        # create a step like EvaluateAllModels
        if self._hyperopt:
            if len(context.model_space) > 1:
                raise ValueError("Multiple models are not supported by Hyperopt yet")
            model = context.model_space[0]

            if model_params is not None:
                model = model(**model_params) # ugly
            cv_score = cross_val_score(
                   model,
                   dataset.data,
                   dataset.target,
                   cv=self._n_folds,
                   n_jobs=self._n_jobs)
            return {'loss': np.mean(cv_score), 'status': STATUS_OK}
        else:
            for model in context.model_space:

                cv_scores = cross_val_score(
                       model,
                       dataset.data,
                       dataset.target,
                       cv=self._n_folds,
                       n_jobs=self._n_jobs)

                cv_results.append((model, np.mean(cv_scores)))

            return cv_results


class Validate:
    def __init__(self, test_size):
        self._test_size = test_size

    def __call__(self, dataset, context):
        return train_test_split(
            dataset.data, 
            dataset.target, 
            test_size=self._test_size, 
            random_state=42)


class ChooseBest:
    def __init__(self, k):
        self._k = k

    def __call__(self, model_scores, context):
        sorted_scores = sorted(model_scores, key=operator.itemgetter(1))
        return sorted_scores[:self._k]
