import pandas as pd

from pandasql.core import DataFrame


def read_csv(*args, name=None, **kwargs):
    return DataFrame(pd.read_csv(*args, **kwargs), name=name)


def read_json(*args, name=None, **kwargs):
    return DataFrame(pd.read_json(*args, **kwargs), name=name)


def read_numpy(*args, name=None, **kwargs):
    return DataFrame(pd.read_numpy(*args, **kwargs), name=name)


def read_pickle(*args, name=None, **kwargs):
    return DataFrame(pd.read_pickle(*args, **kwargs), name=name)
