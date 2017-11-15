import logging
from tqdm import tqdm


class PipelineContext:
    """Global context of a Pipeline"""
    def __init__(self):
        self.epoch = 0
        self.prev_step = None
        self.feature_scores = None
        self.model_space = [] 


class PipelineData:
    def __init__(self, dataset, return_val=None):
        self.dataset = dataset
        self.return_val = return_val

        
class PipelineStep:
    """Base class for all pipeline steps
    
    Example:
        >>> PipelineStep('step name', lambda x: x * 2)
    """

    def __init__(self, name, func, initializer=False, *args, **kwargs):
        self._log = logging.getLogger(self.__class__.__name__)
        self._func = func
        self.name = name
        self._args = args
        self._kwargs = kwargs
        self._initializer = initializer
        self._cached_response = None

    def __call__(self, pipe_input, context):
        if self._initializer:
            if context.epoch == 0:
                self._initializer_was_run = True
                self._cached_response = self._func(pipe_input,
                                                   context,
                                                   *self._args,
                                                   **self._kwargs)
                return self._cached_response
            else:
                return self._cached_response
        else:
            return self._func(pipe_input, context, *self._args, **self._kwargs)

    def is_model_space_functor(self):
        return isinstance(self._func, ModelSpaceFunctor)


class ModelSpaceFunctor:
    """Use this class to mark any PipelineStep to be run on each
    model/parameter set from PipelineContext.model_space"""


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
        if not isinstance(other, PipelineStep):
            if callable(other):
                other = PipelineStep(other.__class__.__name__, other)
            else:
                raise ValueError(("Non-callable step passted to the pipeline."
                                 f"Step {other} must be callable"))

        self.steps.append(other)
        return self

class LocalExecutor:
    """Run AutoML Pipeline locally
    
    Example:
    >>> LocalExecutor(epochs=10) << pipeline
    """

    def __init__(self, input_data=None, *args, **kwargs):
        """
        Parameters
        ----------
        kwargs  
            optional input parameters which will be passed to `run` method,
            useful when you launch pipeline directly via `>>` operator
        """
        self._log = logging.getLogger(self.__class__.__name__)
        self._context = PipelineContext()
        self._args = args
        self._kwargs = kwargs
        self._input_data = input_data

    def __lshift__(self, other):
        return self.run(other, self._input_data, *self._args, **self._kwargs)

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
        if not isinstance(input_data, PipelineData):
            input_data = PipelineData(input_data)

        for epoch_n in range(0, epochs):
            self._context.epoch = epoch_n
            self._log.info(f"Starting AutoML Epoch #{epoch_n + 1}")
            
            pipeline_data = input_data
            for step in tqdm(pipeline.steps):
                self._log.info(f"Running step '{step.name}'")
                if step.is_model_space_functor():
                    # result = []
                    # for model_and_params in self._context.model_space:
                    #     step_return = step(pipeline_data, model_and_params)
                    #     result.append(step_return)
                    #     print('='*20)
                    #     print(f"{step.name}, {model_and_params}")
                    #     print(f"RESULT - {step_return.model}")
                        
                    #     print('='*20)
                    
                    # pipeline_data.return_val = result
                    pipeline_data.return_val = [step(pipeline_data, model_and_params)
                                  for model_and_params in self._context.model_space]
                    
                else:
                    pipeline_data = step(pipeline_data, self._context)
        
        return self._context, pipeline_data
