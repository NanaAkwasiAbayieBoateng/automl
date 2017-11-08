import unittest
from automl.pipeline import LocalExecutor, Pipeline, PipelineStep
from automl.data.dataset import Dataset
from automl.model import ModelSpace, Validate, CV, ChooseBest

from sklearn import datasets
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score


class Data:
    def __init__(self):
        pass

    def __call__(self, pipe_input, context):
        return Dataset(datasets.load_iris().data, datasets.load_iris().target)


class TestSearchPipeline(unittest.TestCase):
    def test_step_validate(self):
        model_list = [
            LogisticRegression(),
            RandomForestClassifier(n_estimators=100),
            GradientBoostingClassifier(),
            SVC(),
            KNeighborsClassifier(),
        ]
        try:
            LocalExecutor() << (Pipeline() >>
                PipelineStep('data', Data()) >>
                PipelineStep('model space', ModelSpace(model_list)) >>
                PipelineStep('validation', Validate(test_size=0.33, metrics=accuracy_score)) >>
                PipelineStep('choose', ChooseBest(3)))
        except:
            self.fail("LocalExecutor failed unexpectedly!")

    def test_step_cv(self):
        model_list = [
            LogisticRegression(),
            RandomForestClassifier(n_estimators=100),
            GradientBoostingClassifier(),
            SVC(),
            KNeighborsClassifier(),
        ]
        try:
            LocalExecutor() << (Pipeline() >> 
                PipelineStep('data', Data()) >>
                PipelineStep('model space', ModelSpace(model_list)) >>
                PipelineStep('cv', CV()) >>
                PipelineStep('choose', ChooseBest(3)))
        except:
            self.fail("LocalExecutor failed unexpectedly!")
