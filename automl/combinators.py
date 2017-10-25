import logging
import random
from automl.pipeline import PipelineStep


class RandomChoice:
    """Combinator that chooses one of the steps uniformly at random.
    >>> pipeline >> RandomChoice([step1, step2])
    """

    def __init__(self, steps):
        """Initialize combinator.

        Parameters
        ----------
        steps: iterable of PipelineStep
           steps to choose from 
        """
        self.name = self.__class__.__name__
        self._steps = steps

    def __call__(self, data, context):
        step_choice = random.randint(0, len(self._steps) - 1)
        return self._steps[step_choice](data, context)
