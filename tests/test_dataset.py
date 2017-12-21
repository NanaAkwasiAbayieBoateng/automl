import unittest
import pandas as pd

from automl.data.dataset import Dataset, DatasetExtractor
from automl.pipeline import PipelineContext


class TestDataset(unittest.TestCase):
    def test_dataset_extractor(self):
        test_context = PipelineContext()
        test_data = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
        extractor = DatasetExtractor(target='d')
        dataset = extractor(test_data, test_context)
        self.assertIsInstance(dataset, Dataset)
        self.assertEqual(dataset.target.name, 'd')
        self.assertTrue((dataset.target == [4]).all())
        self.assertTrue((dataset.data == [[1, 2, 3]]).all())

    def test_dataset_extractor_callable_filter(self):
        def col_filter(data):
            return ['a']

        test_context = PipelineContext()
        test_data = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
        extractor = DatasetExtractor(target='d', data_col_filter=col_filter)
        dataset = extractor(test_data, test_context)
        self.assertIsInstance(dataset, Dataset)
        #self.assertTrue((dataset.data.columns == ['a']).all())

    def test_dataset_extractor_filter(self):
        test_context = PipelineContext()
        test_data = pd.DataFrame([[1, 2, 3, 4]], columns=['a', 'b', 'c', 'd'])
        extractor = DatasetExtractor(target='d', data_col_filter=['a'])
        dataset = extractor(test_data, test_context)
        self.assertIsInstance(dataset, Dataset)
        #self.assertTrue((dataset.data.columns == ['a']).all())
