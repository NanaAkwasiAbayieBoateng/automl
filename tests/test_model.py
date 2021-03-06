import unittest

from automl.pipeline import PipelineContext, PipelineData
from automl.data.dataset import Dataset
from automl.model import ModelSpace, CV

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn import datasets
from sklearn.cross_validation import cross_val_score


class TestModel(unittest.TestCase):
    def test_model_space(self):
        model_list_1 = [
            (LogisticRegression, {}),
            (RandomForestClassifier, {}),
            (GradientBoostingClassifier, {}),
            (SVC, {}),
            (KNeighborsClassifier, {}),
        ]

        context = PipelineContext()
        model_space = ModelSpace(model_list_1)
        model_space('data', context)
        self.assertListEqual(model_list_1, context.model_space)

    def test_cv(self):
        dataset = PipelineData(Dataset(datasets.load_iris().data,
                          datasets.load_iris().target))

        cv = CV('accuracy', n_folds=5)
        self.assertAlmostEqual(
                cv(dataset, (RandomForestClassifier, {'random_state': 1})).score,
            cross_val_score(
                RandomForestClassifier(random_state=1),
                dataset.dataset.data,
                dataset.dataset.target,
                cv=5).mean())
