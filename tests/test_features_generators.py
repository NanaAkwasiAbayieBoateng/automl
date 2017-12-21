import unittest
from unittest.mock import Mock
import random

from automl.feature.generators import SklearnFeatureGenerator, FormulaFeatureGenerator, \
RecoveringFeatureGenerator, PolynomialGenerator, PolynomialFeatureGenerator
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
        self.assertTrue((Transformer.fit_transform.call_args[0][0] == df.as_matrix()).all())

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

    def test_recovering_dataset_FFG(self):
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
            >> ChooseBest(1) 
            >> FeatureSelector(30))

        rec = RecoveringFeatureGenerator()
        final_data = rec(pipeline_data.dataset.meta, datasets.load_boston().data)
        self.assertEqual(pipeline_data.dataset.data.shape, final_data.shape)
        self.assertTrue((final_data == pipeline_data.dataset.data).all())

    def test_poly_gen(self):
        model_list = [
            (Lasso, {}),
            #(Ridge, {}),
            (RandomForestRegressor, {})
        ]

        X, y = datasets.make_regression(n_features=5)

        data = Dataset(X, y)
        context, pipeline_data = LocalExecutor(data, 10) << (Pipeline() 
            >> ModelSpace(model_list) 
            >> PolynomialFeatureGenerator(max_degree=3)
            >> Validate(test_size=0.33, metrics=mean_squared_error) 
            >> ChooseBest(1) 
            >> FeatureSelector(10)
            )

        rec = RecoveringFeatureGenerator()
        final_data = rec(pipeline_data.dataset.meta, X.astype('float32'))
        self.assertEqual(pipeline_data.dataset.data.shape, final_data.shape)
        #self.assertTrue((np.around(final_data, decimals=5) == np.around(pipeline_data.dataset.data, decimals=5)).all())
