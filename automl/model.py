import multiprocessing
import operator
import numpy as np

from sklearn.cross_validation import cross_val_score
from sklearn.cross_validation import train_test_split

from automl.pipeline import ModelSpaceFunctor


class ModelSpace:
    """
    Class contains predefined models with given hyperparameters

    Parameters
    ----------
    model_list : list of tuples
        each element must be a tuple where the first element is an estimator
        and the second one is estimator parameter kwargs dict 
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


class CV(ModelSpaceFunctor):
    """
    Class for cross-validation step in pipeline
    """
    def __init__(self, n_folds=3, n_jobs=None):
        self._n_folds = n_folds

        if n_jobs is None:
            self._n_jobs = multiprocessing.cpu_count()
        else:
            self._n_jobs = n_jobs

    def __call__(self, dataset, context, hparams=None):
        model, model_params = context 

        # still a bit ugly...
        # 1. model_params contain hyperopt template if we are using Hyperopt
        # 2. hparams is passsed by hyperopy only when using Hyperopt step
        if hparams is not None:
            model = model(**hparams)
        else:
            model = model(**model_params) 

        self.cv_score = cross_val_score(
               model,
               dataset.data,
               dataset.target,
               cv=self._n_folds,
               n_jobs=self._n_jobs)

        return dataset, (model, np.mean(self.cv_score))


class Validate(ModelSpaceFunctor):
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

    def __call__(self, dataset, context, hparams=None):
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
        model, model_params = context 

        # still a bit ugly...
        # 1. model_params contain hyperopt template if we are using Hyperopt
        # 2. hparams is passsed by hyperopy only when using Hyperopt step
        if hparams is not None:
            model = model(**hparams)
        else:
            model = model(**model_params) 

        X_train, X_test, y_train, y_test = train_test_split(
            dataset.data,
            dataset.target,
            test_size=self._test_size,
            random_state=42)

        model.fit(X_train, y_train)
        validation_results = self._metrics(model.predict(X_test), y_test)
        return dataset, (model, validation_results)


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
        model_scores = [inp[1] for inp in pipe_input]
        sorted_scores = sorted(model_scores, key=operator.itemgetter(1))
        return sorted_scores[:self._k]
