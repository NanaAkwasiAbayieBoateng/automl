import unittest
import sklearn
import pandas as pd
import numpy as np

from automl.pipeline import PipelineStep, Pipeline, LocalExecutor, PipelineData
from automl.combinators import RandomChoice
from automl.feature.generators import PolynomialGenerator
from automl.data.dataset import Dataset

from sklearn.preprocessing import PolynomialFeatures

class TestPipeline(unittest.TestCase):
    def test_pipeline_step(self):
        pipeline = Pipeline() >> PipelineStep('a', lambda x, context: PipelineData(x.dataset + 1)) \
                              >> PipelineStep('b', lambda x, context: PipelineData(x.dataset + 2))

        executor = LocalExecutor()
        context, data = executor.run(pipeline, 0)
        self.assertEqual(data.dataset, 3)

    def test_random_choice_combinator(self):
        for _ in range(0, 10):
            data = Dataset(np.zeros((2, 2)), np.zeros((2, 2)))
            result = LocalExecutor() << (Pipeline() >> RandomChoice([
                PipelineStep('a', lambda x, context: 'a'),
                PipelineStep('b', lambda x, context: 'b')
                ]))

            print(result)
            self.assertIn(result[1], ['a', 'b'])

    def test_pipeline(self):
        df = pd.DataFrame([[1, 2], [3, 4]])
        X = Dataset(df, None)
        poly = PolynomialGenerator(interaction_only=True, degree=4)
        context, pipe_output = LocalExecutor(X) << (Pipeline() >> 
                                      PipelineStep('generate_features', poly))

        self.assertEqual(pipe_output.dataset.data.shape, (2, 4))

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
