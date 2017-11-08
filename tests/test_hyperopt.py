import logging
import unittest

import hyperopt

from automl.data.dataset import Dataset
from automl.hyperparam.hspace import random_forest_hp_space
from automl.hyperparam.hyperopt import Hyperopt
from automl.model import CV, ModelSpace
from automl.pipeline import LocalExecutor, Pipeline
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier


class TestHyperparameters(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        logging.basicConfig(level=logging.INFO)

    def test_hyperopt(self):
        max_evals = 2
        x, y = make_classification()
        dataset = Dataset(x, y)
        result = LocalExecutor(dataset) << (
                                    Pipeline() 
                                    >> ModelSpace([(RandomForestClassifier, random_forest_hp_space())])
                                    >> Hyperopt(CV(), 
                                                max_evals=max_evals)
                                    )
        dataset, trials = result[1][0] 
        self.assertIsInstance(trials, hyperopt.base.Trials) 
        self.assertEqual(len(trials), max_evals) 
