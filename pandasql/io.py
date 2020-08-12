import os
import subprocess

import pandas as pd

from pandasql.core import DB_FILE, SQL_CON, DataFrame, _new_name

SAMPLE_LINES = 1000
MEMORY_THRESHOLD = 4_000_000_000    # 4 GB
CHUNKSIZE = 10_000


def read_csv(*args, name=None, **kwargs):

    if 'nrows' in kwargs:
        raise ValueError('nrows is not supported')

    name = name or _new_name()
    sql_load = kwargs.pop('sql_load') if 'sql_load' in kwargs else False

    if sql_load:    # read through sqlite
        return _csv_to_sql(*args, name=name, **kwargs)

    estimated_size = _estimate_pd_memory(args[0])

    if estimated_size > MEMORY_THRESHOLD:    # TODO: set auto threshold
        return _csv_chunking(*args, name=name, **kwargs)

    return DataFrame(pd.read_csv(*args, **kwargs), name=name, offload=True,
                     loaded_on_sqlite=False)


def read_json(*args, name=None, **kwargs):
    return DataFrame(pd.read_json(*args, **kwargs), name=name)


def read_numpy(*args, name=None, **kwargs):
    return DataFrame(pd.read_numpy(*args, **kwargs), name=name)


def read_pickle(*args, name=None, **kwargs):
    return DataFrame(pd.read_pickle(*args, **kwargs), name=name)


def _csv_to_sql(*args, name, **kwargs):
    kwargs['nrows'] = SAMPLE_LINES
    chunk = pd.read_csv(*args, **kwargs)
    chunk.to_sql(name=name, con=SQL_CON, index=False, if_exists='append')
    subprocess.call(["sqlite3", DB_FILE, ".mode csv",
                     f".import  \'| tail -n +{SAMPLE_LINES + 2} {args[0]}\' {name}"])   # noqa

    df = pd.DataFrame(columns=chunk.columns)
    return DataFrame(df, name=name, offload=False, loaded_on_sqlite=True)


def _csv_chunking(*args, name, **kwargs):
    kwargs['chunksize'] = CHUNKSIZE
    kwargs['nrows'] = None
    for chunk in pd.read_csv(*args, **kwargs):
        chunk.to_sql(name=name, con=SQL_CON, index=False, if_exists='append')
        cols = chunk.columns

    df = pd.DataFrame(columns=cols)
    return DataFrame(df, name=name, offload=False, loaded_on_sqlite=True)


##############################################################################
#                           Utility Functions
##############################################################################
def _estimate_pd_memory(*args, **kwargs):
    kwargs['nrows'] = SAMPLE_LINES
    df = pd.read_csv(*args, **kwargs)
    temp = ".csv_topn"
    filename = args[0]

    if isinstance(df, pd.Series):
        sample_memory_usage = df.memory_usage(deep=True)
    else:
        sample_memory_usage = df.memory_usage(deep=True).sum()

    with open(temp, "w+") as f:
        subprocess.call(["head", "-n", str(SAMPLE_LINES), filename], stdout=f)

    sample_disk_size = os.path.getsize(temp)
    # os.remove(temp)

    full_disk_size = os.path.getsize(filename)

    est_memory_size = (full_disk_size / sample_disk_size) * sample_memory_usage

    return est_memory_size
