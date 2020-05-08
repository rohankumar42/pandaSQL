import pandas as pd

import pandasql as ps


def read_csv(*args, name=None, **kwargs):
    return ps.DataFrame(pd.read_csv(*args, **kwargs), name=name)


def read_csvs(files, *args, name=None, **kwargs):
    return ps.concat([read_csv(f, *args, name=name, **kwargs)
                      for f in files])


def read_json(*args, name=None, **kwargs):
    return ps.DataFrame(pd.read_json(*args, **kwargs), name=name)


def read_numpy(*args, name=None, **kwargs):
    return ps.DataFrame(pd.read_numpy(*args, **kwargs), name=name)


def read_pickle(*args, name=None, **kwargs):
    return ps.DataFrame(pd.read_pickle(*args, **kwargs), name=name)
