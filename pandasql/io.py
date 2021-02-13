import subprocess

import pandas as pd
from pandasql.core import DB_FILE, SQL_CON, DataFrame, _new_name
from pandasql.memory_utils import _estimate_pandas_memory_from_csv, \
    _free_memory

SAMPLE_LINES = 1000
CHUNKSIZE = 10_000


def read_csv(file_name, name=None, sql_load=False, **kwargs):

    if 'nrows' in kwargs:
        raise ValueError('nrows is not supported')

    name = name or _new_name()

    if sql_load:    # read through SQLite
        return _csv_to_sqlite(file_name, name=name, **kwargs)

    estimated_size = _estimate_pandas_memory_from_csv(file_name, **kwargs)

    if estimated_size > _free_memory():
        return _read_csv_by_chunking(file_name, name=name, **kwargs)

    return DataFrame(pd.read_csv(file_name, **kwargs), name=name,
                     offload=True, loaded_on_sqlite=False)


def read_json(*args, name=None, **kwargs):
    return DataFrame(pd.read_json(*args, **kwargs), name=name)


def read_numpy(*args, name=None, **kwargs):
    return DataFrame(pd.read_numpy(*args, **kwargs), name=name)


def read_pickle(*args, name=None, **kwargs):
    return DataFrame(pd.read_pickle(*args, **kwargs), name=name)


def _csv_to_sqlite(file_name, name, **kwargs):
    chunk = pd.read_csv(file_name, nrows=SAMPLE_LINES, **kwargs)

    # sends first N lines to sqlite to establish correct types
    chunk.to_sql(name=name, con=SQL_CON, index=False, if_exists='append')

    # loads rest of data via direct sqlite CLI call
    cmd = ["sqlite3", DB_FILE, ".mode csv"]
    if kwargs.get('delimiter') is not None:
        cmd += [f".separator \"{kwargs['delimiter']}\""]
    cmd += [f".import  \'| tail -n +{SAMPLE_LINES + 2} {file_name}\' {name}"]
    subprocess.call(cmd)

    df = DataFrame(None, name=name, offload=False, loaded_on_sqlite=True)
    df.columns = chunk.columns
    return df


def _read_csv_by_chunking(file_name, name, **kwargs):
    for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE,
                             nrows=None, **kwargs):
        chunk.to_sql(name=name, con=SQL_CON, index=False, if_exists='append')
        cols = chunk.columns

    df = DataFrame(None, name=name, offload=False, loaded_on_sqlite=True)
    df.columns = cols
    return df
