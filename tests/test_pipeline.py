import unittest
import sklearn
import pandas as pd

from automl.pipeline import PipelineStep, Pipeline, LocalExecutor
from automl.combinators import RandomChoice
from automl.feature.generators import PolynomialGenerator
from automl.data.dataset import Dataset

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
        df = pd.DataFrame([[1, 2], [3, 4]])
        X = Dataset(df, None)
        poly = PolynomialGenerator(interaction_only=True, degree=4)
        result = LocalExecutor(X) << (Pipeline() >> 
                                      PipelineStep('generate_features', poly))

        self.assertEqual(result[1].data.shape, (2, 4))

    def test_initializer(self):
        func = lambda x, context: context.epoch
        result = LocalExecutor(epochs=10) << (Pipeline() 
                                              >> PipelineStep('a',
                                                              func,
                                                              initializer=True))
        self.assertEqual(result[1], 0)

    def test_auto_step_wrapper(self):
        func = lambda x, context: 1
        result = LocalExecutor() << (Pipeline() >> func)
        self.assertEqual(result[1], 1)

    def test_auto_step_wrapper_error(self):
        with self.assertRaises(ValueError):
            LocalExecutor() << (Pipeline() >> "err")
