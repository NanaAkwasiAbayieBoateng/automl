import unittest
from automl.pipeline import PipelineStep, Pipeline, LocalExecutor

class TestPipeline(unittest.TestCase):
    def test_pipeline_step(self):
        pipeline = Pipeline() >> PipelineStep('a', lambda x, context: x + 1) \
                              >> PipelineStep('b', lambda x, context: x + 2) 

        executor = LocalExecutor()
        result = executor.run(pipeline, 0)
        self.assertEqual(result[1], 3)
        
    
             
             

