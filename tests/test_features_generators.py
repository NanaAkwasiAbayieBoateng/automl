import unittest
from unittest.mock import patch

from automl.feature.generators import PolynomialFeatureGenerator, FeatureGeneratorBase


class TestPolynomialFeatureGenerator(unittest.TestCase):
    @patch('sklearn.preprocessing.PolynomialFeatures.fit_transform')
    def test_generate_features(self, fit_transform_mock):
        X=[[1,2],[3,4]]
        generator = PolynomialFeatureGenerator()
        generator.generate_features(X)
        fit_transform_mock.assert_called_with(X)

    