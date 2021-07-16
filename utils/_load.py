import pandas as pd


def load(_path, _columns):

    """
    Load a given dataset.
    Args:
        _path: String
            The path for the dataset.
        _columns:
            The input schema.
            # By default: Organized as {Group attributes, Other attributes, Actionable attributes, Aggregation attribute}
    """

    data = pd.read_csv(_path, header=None)
    data = data[1:].iloc[:, 1:]
    data.columns = _columns
    return data
