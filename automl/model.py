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

        Returns
        -------
        dataset : Dataset
            Unchanged dataset is passed to next step in pipeline
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
        Execute cross-validation for all model in model_space.

        Parametrs
        ---------
        dataset : Dataset
            Processed dataset

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        dataset : Dataset
            Unchanged dataset is passed to next step in pipeline

        cv_results : list of tuples
            Tuples like (model, score)
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

        return dataset, cv_results


class Validate:
    """
    Class for validation step in pipeline with using user metrics
    """

    def __init__(self, test_size, metrics):
        """
        Parametrs
        ---------
        test_size : float, int
            If float, should be between 0.0 and 1.0 and represent the
            proportion of the dataset to include in the test split.
            If int, represents the absolute number of test samples.

        metrics : callable
            Should get predicted and test vector for calculating metrics score
        """
        self._test_size = test_size
        self._metrics = metrics

    def __call__(self, dataset, context):
        """
        Executes validation for all model in model_space.

        Parametrs
        ---------
        dataset : Dataset
            Processed dataset

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        dataset : Dataset
            Unchanged dataset is passed to next step in pipeline

        validation_results : list of tuples
            Tuples like (model, score)
        """
        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=self._test_size,
            random_state=42)
        validation_results = []

        for model in context.model_space:
            model.fit(X_train, y_train)
            validation_results.append((model, self._metrics(
                model.predict(X_test), y_test)))
        return dataset, validation_results


class ChooseBest:
    """
    Chooses best model by scores on CV or Validation step in pipeline
    """

    def __init__(self, k):
        """
        Parametrs
        ---------
        k : int
            Number of models for choice
        """
        self._k = k

    def __call__(self, pipe_input, context):
        """
        Parametrs
        ---------
        dataset : Dataset
            Processed dataset

        model_scorer : list of tuples
            Tuples like (model, score)

        context : PiplineContext
            Global context of pipeline

        Returns
        -------
        dataset : Dataset
            Unchanged dataset is passed to next step in pipeline

        sorted_scores : list of tuples 
            Only the top self._k tuples like (model, score) with the best score
        """
        dataset = pipe_input[0]
        model_scores = pipe_input[1]
        sorted_scores = sorted(model_scores, key=operator.itemgetter(1))
        return dataset, sorted_scores[:self._k]
