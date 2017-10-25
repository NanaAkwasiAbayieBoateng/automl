import unittest
from automl.pipeline import PipelineStep, Pipeline, LocalExecutor
from automl.combinators import RandomChoice


class TestPipeline(unittest.TestCase):
    def test_pipeline_step(self):
        pipeline = Pipeline() >> PipelineStep('a', lambda x, context: x + 1) \
                              >> PipelineStep('b', lambda x, context: x + 2) 

        executor = LocalExecutor()
        result = executor.run(pipeline, 0)
        self.assertEqual(result[1], 3)

    def test_random_choice_combinator(self):
        for _ in range(0, 10):
            result = LocalExecutor() << (Pipeline() >> RandomChoice([
                PipelineStep('a', lambda x, context: 1),
                PipelineStep('b', lambda x, context: 2)
                ]))

            print(result)
            self.assertIn(result[1], [1, 2])
