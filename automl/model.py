import multiprocessing
import operator
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

class ModelSpace:
    """
    Class contains predefined models with given hyperparameters

    Parameters
    ----------
    model_list : list
        List of models
    """
    def __init__(self, model_list):
        self._model_list = model_list

    def __call__(self, dataset, context):
        """
        Sets model space in PipelineContext and passes unchanged dataset on the next step

        Parametrs
        ---------
        dataset : Dataset
            Processed dataset
        
        context : PiplineContext
            Global context of pipeline
        """
        context.model_space = self._model_list
        return dataset

class CV:
    """
    Class for cross-validation step in pipeline
    """
    def __init__(self, n_folds=5, n_jobs=None):
        """
        Parameters
        ----------
        n_folds : int, optional
            Determines the number of cross-validation folds

        n_jobs : integer, optional
            The number of CPUs to use to do the computation. None means ‘all CPUs’.

        """
        self._n_folds = n_folds

        if n_jobs is None:
            self._n_jobs = multiprocessing.cpu_count()
        else:
            self._n_jobs = n_jobs

    def __call__(self, dataset, context):
        """
        
        """
        cv_results = [] 
        
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
    def __init__(self, test_size, metrics):
        self._test_size = test_size
        self._metrics = metrics

    def __call__(self, dataset, context):
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data, 
            dataset.target, 
            test_size=self._test_size, 
            random_state=42)
        validate_results = []

        for model in context.model_space:
            model.fit(X_train, y_train)
            validate_results.append((model, self._metrics(model.predict(X_test), y_test)))
        return validate_results


class ChooseBest:
    def __init__(self, k):
        self._k = k

    def __call__(self, model_scores, context):
        sorted_scores = sorted(model_scores, key=operator.itemgetter(1))
        return sorted_scores[:self._k]
