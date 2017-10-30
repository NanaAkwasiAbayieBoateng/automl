import unittest
import sklearn

from automl.pipeline import PipelineStep, Pipeline, LocalExecutor
from automl.combinators import RandomChoice
from automl.feature.generators import PolynomialGenerator

from sklearn.preprocessing import PolynomialFeatures
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

    def test_pipeline(self):
        X = [[1, 2], [3, 4]]

        poly = PolynomialGenerator(interaction_only=True, degree=4)
        result = LocalExecutor(X) << (Pipeline() >> 
                                      PipelineStep('generate_features', poly))

        self.assertEqual(result[1].shape, (2, 4))
