"""Dataset loaders"""

import logging
from urllib.parse import urlparse

import pandas as pd


class DatasetLoader:
    """Data loading for common formats"""

    def __init__(self, **kwargs):
        self._log = logging.getLogger(self.__class__.__name__)
        self._loader_map = {
                'csv': self._load_csv
        }
        self._loader_args = kwargs

    def __call__(self, resource_uri, pipeline_context=None):
        """Load resource of supported type.

        You can use pandas options for most formats:
        >>> default_dataset_loader("csv:///data.csv", delimeter=";")

        
        Parameters
        ----------
        resource_uri: str
            URI in form of `format:///path/to/resource`
        kwargs:
            keyword arguments are passed to concrete loader implementation


        Returns
        -------
        dataframe: pandas.DataFrame
            resulting data
        """

        uri = urlparse(resource_uri)
        if uri.scheme == '':
            raise ValueError("Please provide format for loading")
        if uri.scheme not in self._loader_map:
            raise ValueError(f"{uri.scheme} is not supported")

        return self._loader_map[uri.scheme](uri.path, **self._loader_args)
    
    def _load_csv(self, csv_path, **kwargs):
        return pd.read_csv(csv_path, **self._loader_args)

    def register_custom_loader(self, format_name, loader_func):
        """Registers custom data loader function.
        
        Parameters
        ----------
        format_name: str
            format name to use as a scheme prefix in URI
        loader_func: callable
            callable that performs data loading
        """
        if not callable(loader_func):
            raise ValueError("loader_func must be callable")
        self._loader_map[format_name] = loader_func
