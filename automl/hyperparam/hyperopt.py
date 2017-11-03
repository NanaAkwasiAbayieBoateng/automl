import logging
from functools import partial
from hyperopt import hp, fmin, tpe, STATUS_OK, Trials


class Hyperopt:
    def __init__(self, score_step, hparams, max_evals=100):
        self._log = logging.getLogger(self.__class__.__name__)
        self._score_step = score_step
        self._hparams = hparams
        self._max_evals = max_evals
        self._trials = Trials()

    def __call__(self, dataset, context):
        score = partial(self._score_step, dataset, context)
        fmin(score,
             space=self._hparams,
             algo=tpe.suggest,
             trials=self._trials,
             max_evals=self._max_evals)
        return self._trials
