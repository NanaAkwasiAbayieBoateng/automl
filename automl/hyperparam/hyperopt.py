import logging
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from automl.pipeline import ModelSpaceFunctor, PipelineData
from hyperopt import STATUS_OK

class HyperparameterSearchResult:
    def __init__(self, best_model, best_score, history=None):
        self.best_model = best_model,
        self.best_score = best_score
        self.history = history


class Hyperopt(ModelSpaceFunctor):
    def __init__(self, score_step_fn, max_evals=100):
        self._log = logging.getLogger(self.__class__.__name__)

        # wrap score function so it returns appropriate format for hyperopt
        self._score_step_fn = lambda *args, **kwargs: {
                'loss': score_step_fn(*args, **kwargs).score,
                'status': STATUS_OK 
                }

        self._max_evals = max_evals
        self._trials = Trials()

    def __call__(self, pipeline_data, context):
        model, hparam_space = context

        score = partial(self._score_step_fn, pipeline_data, context)
        best = fmin(score,
                    space=hparam_space,
                    algo=tpe.suggest,
                    trials=self._trials,
                    max_evals=self._max_evals)
        best_loss = sorted(self._trials.losses())[0]
        result = HyperparameterSearchResult(best,
                                            best_loss,
                                            self._trials)
        return result

