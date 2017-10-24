import unittest
from unittest.mock import patch
from dataset.loader import default_dataset_loader as loader

class TestDataLoader(unittest.TestCase):

    @patch('pandas.read_csv')
    def test_load_csv(self, read_csv_mock):
        loader("csv:///test.csv", delimeter=";")
        read_csv_mock.assert_called_with("/test.csv", delimeter=";")

    def test_unknown_format(self):
        self.assertRaises(ValueError, loader, "notaformat:///path")

    def test_no_format(self):
        self.assertRaises(ValueError, loader, "")
