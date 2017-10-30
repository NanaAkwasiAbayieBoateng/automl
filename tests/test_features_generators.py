import unittest
from unittest.mock import patch, MagicMock, Mock
import numpy as np

from automl.feature.generators import SklearnFeatureGenerator
from automl.pipeline import PipelineContext

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

    def test_generate_features_kwargs(self):
        Transformer = Mock()

        kwargs = {
            'degree': 3
        }
        
        transformer = lambda *args, **kwargs: Transformer(*args, **kwargs)
        gen = SklearnFeatureGenerator(transformer, **kwargs)
        Transformer.assert_called_with(**kwargs)


