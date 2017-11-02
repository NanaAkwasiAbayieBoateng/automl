class Dataset:
    def __init__(self, data, target):
        self.data = data
        self.target = target


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
