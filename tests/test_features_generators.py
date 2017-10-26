import unittest
from unittest.mock import patch
import numpy as np
from sklearn.preprocessing import PolynomialFeatures

from automl.feature.generators import PolynomialFeatureGenerator
from automl.pipeline import PipelineContext


class TestPolynomialFeatureGenerator(unittest.TestCase):
    @patch('sklearn.preprocessing.PolynomialFeatures.fit_transform')
    def test_call_generator(self, fit_transform_mock):
        X = [[1,2],[3,4]]
        context = PipelineContext()
        generator = PolynomialFeatureGenerator()
        generator(X, context)
        fit_transform_mock.assert_called_with(X)

    @patch('automl.feature.generators.PolynomialFeatures')
    def test_generate_features_kwargs(self, features_mock):
        kwargs = {
            'degree': 3
        }
        generator = PolynomialFeatureGenerator(**kwargs)
        features_mock.assert_called_with(**kwargs)