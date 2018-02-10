"""Pipeline steps and general classes for hyperparameter optimization"""

import logging
from functools import partial
from hyperopt import fmin, tpe, STATUS_OK, Trials
from automl.pipeline import ModelSpaceFunctor


class HyperparameterSearchResult:
    """A container for the results of a parameter seach"""

    def __init__(self, best_model, best_score, history=None):
        """Create hyperparameter search resutls.

        Parameters
        ----------
        best_model
            best model found during parameter search
        score
            best score achieved
        history
            object containing all hyperparameter search trials
        """
        self.model = best_model
        self.score = best_score
        self.history = history


class Hyperopt(ModelSpaceFunctor):
    """Pipeline step for executing hyperparameter search with Hyperopt library
    """
    def __init__(self, score_step_fn, max_evals=100, reverse_score=True):
        """Hyperparameter search with Hyperopt
        
        Parameters
        ----------
        score_step_fn: callable
            Funtion that will train given model and return resulting score.
            Can be a CV or Validation step.

        max_evals: int
            Limits the maximum number of search steps

        reverse_score: boolean
            As Hyperopt search parameters that minimize the score sometimes
            it's convenient to reverse it so the optimization proceeds in a
            correct way. For example, when using ROC AUC scorer setting this
            parameter to True will result in sound optimization problem
            formulation.
        """
        self._log = logging.getLogger(self.__class__.__name__)
        
        def score_wrapper(*args, **kwargs):
            """Wrap score function so it returns appropriate format for
            hyperopt"""
            score_result = score_step_fn(*args, **kwargs)
            if reverse_score:
                score = 1 - score_result.score 
            else:
                score = score_result.score

            return {
                'loss': score,
                'status': STATUS_OK,
                'model': score_result.model
                }

        self._score_step_fn = score_wrapper

        self._max_evals = max_evals
        self._reverse_score = reverse_score

    def __call__(self, pipeline_data, context):
        model, hparam_space = context
        self._log.info(hparam_space)
        if not hparam_space:
            self._log.warn((f"Skipping hyperopt step for model {model}. No "
                            "parameter templats found"))
            return HyperparameterSearchResult(model, 0, None)

        trials = Trials()
        self._log.info(f"Running hyperparameter optimization for {model}")

        score = partial(self._score_step_fn, pipeline_data, context)

        fmin(score,
             space=hparam_space,
             algo=tpe.suggest,
             trials=trials,
             max_evals=self._max_evals)

        if self._reverse_score:
            self._log.info("Reversing best score bask to original form as reverse_score=True")
            best_score = 1 - sorted(trials.losses())[0]
        else:
            best_score = sorted(trials.losses())[0]

        best = trials.best_trial['result']['model']
        result = HyperparameterSearchResult(best,
                                            best_score,
                                            trials)
        return result
