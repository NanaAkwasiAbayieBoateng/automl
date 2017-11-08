import logging
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
from automl.pipeline import ModelSpaceFunctor
from hyperopt import STATUS_OK

class Hyperopt(ModelSpaceFunctor):
    def __init__(self, score_step_fn, max_evals=100):
        self._log = logging.getLogger(self.__class__.__name__)

        # wrap score function so it returns appropriate format for hyperopt
        self._score_step_fn = lambda *args, **kwargs: {
                'loss': score_step_fn(*args, **kwargs)[1][1],
                'status': STATUS_OK 
                }

        self._max_evals = max_evals
        self._trials = Trials()

    def __call__(self, dataset, context):
        model, hparam_space = context

        score = partial(self._score_step_fn, dataset, context)
        fmin(score,
             space=hparam_space,
             algo=tpe.suggest,
             trials=self._trials,
             max_evals=self._max_evals)
        return dataset, self._trials

