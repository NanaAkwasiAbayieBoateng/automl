"""Dataset abstraction"""
import numpy as np

from automl.expression import Atom

class Dataset:
    """AutoML Dataset. This class is the main input object that needs to be
    passed into the Pipeline
    """

    def __init__(self, data, target):
        """Create Dataset.

        Parameters
        ----------
        data: pandas.DataFrame or numpy.array
            training data
        target: pandas.DataFrame or numpy.array
            target variable
        """
        if hasattr(data, 'columns'):
            self.meta = []
            for i, col in enumerate(data.columns):
                self.meta.append({"name": col,
                                  "history": Atom(i),
                                  "depth": 1})

        else:
            self.meta = [{"name": f"base_feature_{i}",
                          "history": Atom(i),
                          "depth": 1} for i in range(0, data.shape[1])]

        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype='float32')
        else:
            data = data.astype('float32')
        self.data = data
        self.target = target

    @property
    def columns(self):
        """Get column names

        Returns
        -------
        columns: list[str]
        """
        return [m['name'] for m in self.meta]


class DatasetExtractor:
    def __init__(self, target, data_col_filter=None):
        """Extract Dataset from pandas DataFrame

        Parameters
        ----------
        target: str or int
            target column name
        data_col_filter: list of str, str, or callable
            filter data columns if needed. Can be function that accepts input
            dataset and return a column list
        """
        self._target = target

        self._data_col_filter = data_col_filter

    def __call__(self, x, context):
        data = x.drop(self._target, axis=1)
        if self._data_col_filter is not None:
            if callable(self._data_col_filter):
                col_list = self._data_col_filter(x)
            else:
                col_list = self._data_col_filter

            data = data[col_list]

        return Dataset(data, x[self._target])
