import logging
from tqdm import tqdm


class PipelineContext:
    def __init__(self):
        self.epoch = 0
        self.prev_step = None
        self.feature_scores = None


class PipelineStep:
    def __init__(self, name, func, *args, **kwargs):
        self._log = logging.getLogger(self.__class__.__name__)
        self._func = func
        self.name = name
        self._args = args
        self._kwargs = kwargs

    def __call__(self, pipe_input, context):
        return self._func(pipe_input, context, *self._args, **self._kwargs)


class Pipeline:
    """AutoML Pipeline"""

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
    """Run AutoML Pipeline locally"""

    def __init__(self):
        self._log = logging.getLogger(self.__class__.__name__)
        self._context = PipelineContext()

    def run(self, pipeline, input_data, epochs=1):
        for epoch_n in range(0, epochs):
            self._context.epoch = epoch_n
            self._log.info(f"Starting AutoML Epoch #{epoch_n + 1}")
            
            pipe_output = input_data
            for step in tqdm(pipeline.steps):
                self._log.info(f"Running step '{step.name}'")
                pipe_output = step(pipe_output, self._context)
        
        return self._context, pipe_output

