import unittest
from unittest.mock import patch

from automl.data.loader import DatasetLoader


class TestDataLoader(unittest.TestCase):
    @patch('pandas.read_csv')
    def test_load_csv(self, read_csv_mock):
        loader = DatasetLoader(delimeter=";")
        loader("csv:///test.csv")
        read_csv_mock.assert_called_with("/test.csv", delimeter=";")

    def test_unknown_format(self):
        loader = DatasetLoader()
        self.assertRaises(ValueError, loader, "notaformat:///path")

    def test_no_format(self):
        loader = DatasetLoader()
        self.assertRaises(ValueError, loader, "")

    def test_custom_loader(self):
        loader = DatasetLoader()
        loader.register_custom_loader("custom",
                                      lambda x: "custom_loader_called")

        result = loader("custom:///path")
        self.assertEqual(result, "custom_loader_called")

    def test_custom_loader_err(self):

        loader = DatasetLoader()
        self.assertRaises(ValueError, loader.register_custom_loader, "custom",
                          "")
