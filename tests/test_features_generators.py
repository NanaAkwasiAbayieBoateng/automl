import unittest
from unittest.mock import Mock
import random

from automl.feature.generators import SklearnFeatureGenerator, FormulaFeatureGenerator
from automl.pipeline import PipelineContext
from automl.data.dataset import Dataset

import numpy as np
import pandas as pd


class TestSklearnFeatureGenerator(unittest.TestCase):
    def test_call_generator(self):
        Transformer = Mock()
        Transformer.fit_transform.return_value = []

        df = pd.DataFrame([[1, 2], [3, 4]])
        X = Dataset(df, None)
        context = PipelineContext()

        transformer = lambda *args, **kwargs: Transformer
        gen = SklearnFeatureGenerator(transformer)
        gen(X, context)
        Transformer.fit_transform.assert_called()
        self.assertTrue(Transformer.fit_transform.call_args[0][0].equals(df))

    def test_generate_polynomial_features_kwargs(self):
        Transformer = Mock()

        kwargs = {'degree': 3}

        transformer = lambda *args, **kwargs: Transformer(*args, **kwargs)
        gen = SklearnFeatureGenerator(transformer, **kwargs)
        Transformer.assert_called_with(**kwargs)

    def test_generate_formula_feature(self):
        features = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        df = pd.DataFrame(features)
        X = Dataset(df, None)
        gen = FormulaFeatureGenerator(['+', '*', '/', '-'])
        limit = random.randint(0, 100)
        context = PipelineContext()
        self.assertEqual(
            np.array(features).shape[1] + limit, gen(X, limit, context).data.shape[1])
