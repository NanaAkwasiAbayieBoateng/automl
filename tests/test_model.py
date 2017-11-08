import unittest

from automl.pipeline import PipelineContext
from automl.data.dataset import Dataset
from automl.model import ModelSpace, CV
from automl.data.dataset import Dataset

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
        dataset = Dataset(datasets.load_iris().data,
                          datasets.load_iris().target)

        cv = CV(n_folds=5)
        self.assertEqual(
                cv(dataset, (RandomForestClassifier, {'random_state': 1}))[1][1],
            cross_val_score(
                RandomForestClassifier(random_state=1),
                dataset.data,
                dataset.target,
                cv=5).mean())
