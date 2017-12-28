import unittest
import logging
import random
from automl.pipeline import LocalExecutor, Pipeline, PipelineStep
from automl.data.dataset import Dataset
from automl.model import ModelSpace, Validate, CV, ChooseBest
from automl.feature.selector import FeatureSelector, VotingFeatureSelector, RecursiveFeatureSelector
from automl.feature.generators import FormulaFeatureGenerator, Preprocessing
from automl.hyperparam.hyperopt import Hyperopt
from automl.hyperparam.templates import random_forest_hp_space, knn_hp_space, svc_kernel_hp_space, grad_boosting_hp_space, xgboost_hp_space

from sklearn import datasets
from sklearn.datasets import make_classification, make_regression
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVC, SVR
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, mean_absolute_error, roc_auc_score, mean_squared_error

from xgboost.sklearn import XGBClassifier, XGBRegressor



class IntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)

    def test_step_validate(self):
        model_list = [
            (LogisticRegression, {}),
            (RandomForestClassifier, {'n_estimators': 100}),
            (GradientBoostingClassifier, {}),
            (SVC, {}),
            (KNeighborsClassifier, {}),
        ]

        data = Dataset(datasets.load_iris().data, datasets.load_iris().target)
        LocalExecutor(data) << (Pipeline() >>
            PipelineStep('model space', ModelSpace(model_list)) >>
            PipelineStep('validation', Validate(test_size=0.33, metrics=accuracy_score)) >>
            PipelineStep('choose', ChooseBest(3)))

    def test_step_cv(self):
        model_list = [
            (LogisticRegression, {}),
            (RandomForestClassifier, {'n_estimators': 100}),
            (GradientBoostingClassifier, {}),
            (SVC, {}),
            (KNeighborsClassifier, {}),
        ]

        data = Dataset(datasets.load_iris().data, datasets.load_iris().target)
        LocalExecutor(data) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list)) >>
            PipelineStep('cv', CV('accuracy')) >>
            PipelineStep('choose', ChooseBest(3)))

    def test_step_space_regression_model(self):
        model_list = [
            (Lasso, {}),
            (Ridge, {}),
            (KernelRidge, {}),
        ]

        data = Dataset(datasets.load_boston().data, datasets.load_boston().target)
        LocalExecutor(data) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list)) >>
            PipelineStep('cv', Validate(test_size=0.33, metrics=mean_absolute_error)) >>
            PipelineStep('choose', ChooseBest(3)))

    def test_all_step(self):
        model_list = [
            #(Lasso, {}),
            #(Ridge, {}),
            #(KernelRidge, {}),
            (RandomForestRegressor, {}),
            (XGBRegressor, {})
        ]

        data = Dataset(datasets.load_boston().data, datasets.load_boston().target)
        context, pipeline_data = LocalExecutor(data, 10) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list)) >>
            PipelineStep('feature generation', FormulaFeatureGenerator(['+', '-', '*'])) >>
            PipelineStep('cv', Validate(test_size=0.33, metrics=mean_squared_error)) >>
            PipelineStep('choose', ChooseBest(1, by_largest_score=False)) >>
            PipelineStep('selection', FeatureSelector(20)))

        print('0'*30)
        for result in pipeline_data.return_val:
            print(result.model, result.score)
        print(pipeline_data.dataset.data.shape)
        print('0'*30)


    def test_pipeline_hyperopt(self):
        x, y = make_classification(
            n_samples=100,
            n_features=40,
            n_informative=2,
            n_redundant=10,
            flip_y=0.05)
        model_list = [
            (RandomForestClassifier, random_forest_hp_space()),
                (GradientBoostingClassifier, grad_boosting_hp_space()),
            (SVC, svc_kernel_hp_space('rbf')),
            (KNeighborsClassifier, knn_hp_space()),
            (XGBClassifier, xgboost_hp_space())
        ]

        data = Dataset(x, y)
        context, pipeline_data = LocalExecutor(data, 2) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list), initializer=True) >>
            PipelineStep('feature generation', FormulaFeatureGenerator(['+', '-', '*'])) >>
            PipelineStep('H', Hyperopt(Validate(test_size=0.1, metrics=roc_auc_score), 
                                                max_evals=2)) >>
            PipelineStep('choose', ChooseBest(1)) 
            >> PipelineStep('selection', FeatureSelector(10))
            )

        print('0'*30)
        for result in pipeline_data.return_val:
            print(result.model, result.score)
        print(pipeline_data.dataset.data.shape)
        print('0'*30)

    def test_voting_feature_selector(self):
        x, y = make_regression(
            n_samples=100,
            n_features=40,
            n_informative=2,
        )

        model_list = [
            (RandomForestRegressor, {}),
            (GradientBoostingRegressor, {}),
            (SVR, {}),
            (XGBRegressor, {})
        ]

        data = Dataset(x, y)

        result_mult = []
        result_div = []
        context, pipeline_data = LocalExecutor(data, 10) << (Pipeline()
            >> PipelineStep('model space', ModelSpace(model_list), initializer=True)
            >> FormulaFeatureGenerator(['+', '-', '*', '/'])
            >> Validate(test_size=0.1, metrics=mean_absolute_error)
            >> ChooseBest(4, by_largest_score=False)
            >> VotingFeatureSelector(feature_to_select=10, reverse_score=True)
        )

        preprocessing = Preprocessing()
        final_data = preprocessing.reproduce(pipeline_data.dataset, Dataset(x, y))
        self.assertEqual(pipeline_data.dataset.data.shape, final_data.shape)
        self.assertTrue((final_data == pipeline_data.dataset.data).all())

        print('0'*30)
        for result in pipeline_data.return_val:
            print(result.model, result.score)
        print(pipeline_data.dataset.data.shape)
        print('0'*30)

    def test_RFS(self):
        x, y = make_classification(
            n_samples=100,
            n_features=40,
            n_informative=2,
            n_redundant=10,
            flip_y=0.05
        )

        model_list = [
            (RandomForestClassifier, {}),
            (GradientBoostingClassifier, {}),
            (SVC, {}),
            (KNeighborsClassifier, {}),
            (XGBClassifier, {})
        ]

        n_features_to_select = random.randint(5, 30)

        data = Dataset(x, y)
        context, pipeline_data = LocalExecutor(data, 2) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list), initializer=True) >>
            PipelineStep('feature generation', FormulaFeatureGenerator(['+', '-', '*', '/'])) >>
            PipelineStep('Validate', Validate(test_size=0.1, metrics=roc_auc_score)) >>
            PipelineStep('choose', ChooseBest(1)) >>
            PipelineStep('selection', RecursiveFeatureSelector(n_features_to_select=n_features_to_select))
        )

        preprocessing = Preprocessing()
        final_data = preprocessing.reproduce(pipeline_data.dataset, Dataset(x, y))
        self.assertEqual(pipeline_data.dataset.data.shape, final_data.shape)
        self.assertTrue((final_data == pipeline_data.dataset.data).all())
