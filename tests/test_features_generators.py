import unittest
from unittest.mock import Mock
import random

from automl.feature.generators import SklearnFeatureGenerator, FormulaFeatureGenerator
from automl.pipeline import PipelineContext

import numpy as np


class TestSklearnFeatureGenerator(unittest.TestCase):
    def test_call_generator(self):
        Transformer = Mock()
        Transformer.fit_transform.return_value = []

        X = [[1, 2], [3, 4]]
        context = PipelineContext()

        transformer = lambda *args, **kwargs: Transformer
        gen = SklearnFeatureGenerator(transformer)
        gen(X, context)
        Transformer.fit_transform.assert_called_with(X)

    def test_generate_polynomial_features_kwargs(self):
        Transformer = Mock()

        kwargs = {'degree': 3}

        transformer = lambda *args, **kwargs: Transformer(*args, **kwargs)
        gen = SklearnFeatureGenerator(transformer, **kwargs)
        Transformer.assert_called_with(**kwargs)

    def test_generate_formula_feature(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9]]
        gen = FormulaFeatureGenerator(['+', '*', '/', '-'])
        limit = random.randint(0, 100)
        context = PipelineContext()
        self.assertEqual(
            np.array(a).shape[1] + limit, gen(a, limit, context).shape[1])
