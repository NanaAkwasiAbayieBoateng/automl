import unittest
from automl.pipeline import LocalExecutor, Pipeline, PipelineStep
from automl.data.dataset import Dataset
from automl.model import ModelSpace, Validate, CV, ChooseBest
from automl.feature.selector import FeatureSelector
from automl.feature.generators import FormulaFeatureGenerator

from sklearn import datasets
from sklearn.linear_model import LogisticRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.kernel_ridge import KernelRidge
from sklearn.metrics import accuracy_score, mean_absolute_error


class TestSearchPipeline(unittest.TestCase):
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
            (Lasso, {}),
            (Ridge, {}),
            (KernelRidge, {}),
        ]

        data = Dataset(datasets.load_boston().data, datasets.load_boston().target)
        context, pipeline_data = LocalExecutor(data, 10) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list)) >>
            PipelineStep('feature generation', FormulaFeatureGenerator(['+', '-', '*'])) >>
            PipelineStep('cv', Validate(test_size=0.33, metrics=mean_absolute_error)) >>
            PipelineStep('choose', ChooseBest(3)) >>
            PipelineStep('selection', FeatureSelector(30)))

        print('0'*30)
        for result in pipeline_data.return_val:
            print(result.model, result.score)
        print(pipeline_data.dataset.data.shape)
        print('0'*30)
