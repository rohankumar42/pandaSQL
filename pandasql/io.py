import os
import subprocess
from tempfile import mkstemp

import pandas as pd

from pandasql.core import DB_FILE, SQL_CON, DataFrame, _new_name

SAMPLE_LINES = 1000
MEMORY_THRESHOLD = 4_000_000_000    # 4 GB
CHUNKSIZE = 10_000


def read_csv(file_name, name=None, sql_load=False, **kwargs):

    if 'nrows' in kwargs:
        raise ValueError('nrows is not supported')

    name = name or _new_name()

    if sql_load:    # read through sqlite
        return _csv_to_sql(file_name, name=name, **kwargs)

    estimated_size = _estimate_pd_memory(file_name)

    if estimated_size > MEMORY_THRESHOLD:    # TODO: set auto threshold
        return _csv_chunking(file_name, name=name, **kwargs)

    return DataFrame(pd.read_csv(file_name, **kwargs), name=name,
                     offload=True, loaded_on_sqlite=False)


def read_json(*args, name=None, **kwargs):
    return DataFrame(pd.read_json(*args, **kwargs), name=name)


def read_numpy(*args, name=None, **kwargs):
    return DataFrame(pd.read_numpy(*args, **kwargs), name=name)


def read_pickle(*args, name=None, **kwargs):
    return DataFrame(pd.read_pickle(*args, **kwargs), name=name)


def _csv_to_sql(file_name, name, **kwargs):
    chunk = pd.read_csv(file_name, nrows=SAMPLE_LINES, **kwargs)

    # sends first N lines to sqlite to establish correct types
    chunk.to_sql(name=name, con=SQL_CON, index=False, if_exists='append')

    # loads rest of data via direct sqlite CLI call
    subprocess.call(["sqlite3", DB_FILE, ".mode csv",
                     f".import  \'| tail -n +{SAMPLE_LINES + 2} {file_name}\' {name}"])   # noqa

    df = DataFrame(None, name=name, offload=False, loaded_on_sqlite=True)
    df.columns = chunk.columns
    return df


def _csv_chunking(file_name, name, **kwargs):
    for chunk in pd.read_csv(file_name, chunksize=CHUNKSIZE,
                             nrows=None, **kwargs):
        chunk.to_sql(name=name, con=SQL_CON, index=False, if_exists='append')
        cols = chunk.columns

    df = DataFrame(None, name=name, offload=False, loaded_on_sqlite=True)
    df.columns = cols
    return df


##############################################################################
#                           Utility Functions
##############################################################################
def _estimate_pd_memory(file_name, **kwargs):
    kwargs['nrows'] = SAMPLE_LINES
    df = pd.read_csv(file_name, **kwargs)
    temp = mkstemp(".csv_topn")[1]

    if isinstance(df, pd.Series):
        sample_memory_usage = df.memory_usage(deep=True)
    else:
        sample_memory_usage = df.memory_usage(deep=True).sum()

    with open(temp, "w+") as f:
        subprocess.call(["head", "-n", str(SAMPLE_LINES), file_name], stdout=f)

    sample_disk_size = os.path.getsize(temp)
    os.remove(temp)

    full_disk_size = os.path.getsize(file_name)

    est_memory_size = (full_disk_size / sample_disk_size) * sample_memory_usage

    return est_memory_size
