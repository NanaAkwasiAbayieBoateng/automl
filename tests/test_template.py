import unittest

from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn import datasets
from sklearn.metrics import mean_absolute_error

from automl.hyperparam.optimization import Hyperopt
from automl.hyperparam.templates import random_forest_hp_space
from automl.pipeline import PipelineStep, LocalExecutor, Pipeline
from automl.model import Validate, ModelSpace
from automl.data.dataset import Dataset

class TestTemplate(unittest.TestCase):
    def test_forest(self):
        model_list = [
            #(RandomForestRegressor, random_forest_hp_space('mae')),
            (RandomForestClassifier, random_forest_hp_space())
        ]

        data = Dataset(datasets.load_iris().data, datasets.load_iris().target)
        context, pipeline_data = LocalExecutor(data, 1) << (Pipeline() >> 
            PipelineStep('model space', ModelSpace(model_list)) >>
            PipelineStep('H', Hyperopt(Validate(test_size=0.33, metrics=mean_absolute_error), max_evals=2)))
