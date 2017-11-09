"""Feature Selector"""

from automl.pipeline import PipelineData

from sklearn.feature_selection import SelectFromModel
from sklearn.utils import safe_mask

import numpy as np

class FeatureSelector:
    def __init__(self):
        pass

    def __call__(self, pipeline_data, context):
        """
        """
        mask = np.array([False for _ in range(0, pipeline_data.dataset.data.shape[1])])
        for value in pipeline_data.return_val:
            model = value.model
            selector = SelectFromModel(model, "0.05*mean", True)
            mask = selector.get_support() + mask
        pipeline_data.dataset.data = pipeline_data.dataset.data[:, mask]
        return PipelineData(pipeline_data.dataset, pipeline_data.return_val)