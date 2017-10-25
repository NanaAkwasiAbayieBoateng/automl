import logging
from tqdm import tqdm


class PipelineContext:
    """Global context of a Pipeline"""
    def __init__(self):
        self.epoch = 0
        self.prev_step = None
        self.feature_scores = None


class PipelineStep:
    """Base class for all pipeline steps
    
    Example:
        >>> PipelineStep('step name', lambda x: x * 2)
    """

    def __init__(self, name, func, *args, **kwargs):
        self._log = logging.getLogger(self.__class__.__name__)
        self._func = func
        self.name = name
        self._args = args
        self._kwargs = kwargs

    def __call__(self, pipe_input, context):
        return self._func(pipe_input, context, *self._args, **self._kwargs)


class Pipeline:
    """AutoML Pipeline
    
    Example:
        >>> Pipeline() >> PipelineStep('hello step', lambda: print('Hi!'))
    """

    def __init__(self, steps=None):
        self._log = logging.getLogger(self.__class__.__name__)
        if steps is not None:
            self.steps = steps
        else:
            self.steps = []
    
    def __rshift__(self, other):
        self.steps.append(other)
        return self

class LocalExecutor:
    """Run AutoML Pipeline locally
    
    Example:
    >>> LocalExecutor(epochs=10) << pipeline
    """

    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        kwargs  
            optional input parameters which will be passed to `run` method,
            useful when you launch pipeline directly via `>>` operator
        """
        self._log = logging.getLogger(self.__class__.__name__)
        self._context = PipelineContext()
        self._kwargs = kwargs

    def __lshift__(self, other):
        return self.run(other, **self._kwargs)

    def run(self, pipeline, input_data=None, epochs=1):
        """Run pipeline.
        
        Parameters
        ----------
        pipeline: Pipeline
        input_data
            inuput to be passed to the pipeline
        epochs: int
            run pipeline a given number of epochs

        Returns
        -------
        result
            result of a given pipeline"""
        for epoch_n in range(0, epochs):
            self._context.epoch = epoch_n
            self._log.info(f"Starting AutoML Epoch #{epoch_n + 1}")
            
            pipe_output = input_data
            for step in tqdm(pipeline.steps):
                self._log.info(f"Running step '{step.name}'")
                pipe_output = step(pipe_output, self._context)
        
        return self._context, pipe_output
