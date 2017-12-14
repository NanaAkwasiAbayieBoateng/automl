import unittest
from unittest.mock import Mock
import random

from automl.feature.generators import SklearnFeatureGenerator, FormulaFeatureGenerator, RecoveringFeatureGenerator
from automl.pipeline import PipelineContext, PipelineData, Pipeline, LocalExecutor
from automl.data.dataset import Dataset
from automl.model import Validate, ModelSpace, ChooseBest
from automl.feature.selector import FeatureSelector 

from sklearn import datasets
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge, Lasso
from sklearn.metrics import mean_squared_error

import numpy as np
import pandas as pd


class TestSklearnFeatureGenerator(unittest.TestCase):
    def test_call_generator(self):
        Transformer = Mock()
        Transformer.fit_transform.return_value = []

        df = pd.DataFrame([[1, 2], [3, 4]])
        X = PipelineData(Dataset(df, None))
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
        X = PipelineData(Dataset(df, None))
        gen = FormulaFeatureGenerator(['+', '*', '/', '-'])
        limit = random.randint(0, 100)
        context = PipelineContext()
        result_size = gen(X, context, limit).dataset.data.shape[1]
        self.assertLessEqual(result_size,
                             np.array(features).shape[1] + limit)

    def test_recovering_dataset(self):
        model_list = [
            (Lasso, {}),
            (Ridge, {}),
            (RandomForestRegressor, {})
        ]

        data = Dataset(datasets.load_boston().data, datasets.load_boston().target)
        context, pipeline_data = LocalExecutor(data, 10) << (Pipeline() 
            >> ModelSpace(model_list) 
            >> FormulaFeatureGenerator(['+', '-', '*']) 
            >> Validate(test_size=0.33, metrics=mean_squared_error) 
            >> ChooseBest(3) 
            >> FeatureSelector(30))

        rec = RecoveringFeatureGenerator()
        pipeline_data_rec = rec(pipeline_data, context)
        self.assertEqual(pipeline_data.dataset.data.shape, pipeline_data_rec.dataset.data.shape)
        self.assertTrue((pipeline_data_rec.dataset.data == pipeline_data.dataset.data).all())
