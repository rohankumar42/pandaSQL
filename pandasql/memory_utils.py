import os
import subprocess
import psutil
from tempfile import mkstemp

import pandas as pd
import pandasql as ps


SAMPLE_ROWS = 1000
# What proportion of available memory should we actually consider available
SAFETY_FACTOR = 0.8


def _estimate_pandas_memory_from_csv(file_name, **kwargs):
    kwargs['nrows'] = SAMPLE_ROWS
    df = pd.read_csv(file_name, **kwargs)
    temp = mkstemp(".csv_topn")[1]

    if isinstance(df, pd.Series):
        sample_memory_usage = df.memory_usage(deep=True)
    else:
        sample_memory_usage = df.memory_usage(deep=True).sum()

    with open(temp, "w+") as f:
        subprocess.call(["head", "-n", str(SAMPLE_ROWS), file_name], stdout=f)

    sample_disk_size = os.path.getsize(temp)
    os.remove(temp)

    full_disk_size = os.path.getsize(file_name)

    est_memory_size = (full_disk_size / sample_disk_size) * sample_memory_usage

    return est_memory_size


def _estimate_pandas_memory_from_sqlite(table_name):
    sample = pd.read_sql(f'SELECT * FROM {table_name} LIMIT {SAMPLE_ROWS}',
                         con=ps.core.SQL_CON)
    stats = pd.read_sql('SELECT SUM(ncell) as nrows FROM dbstat '
                        f'WHERE name="{table_name}"', con=ps.core.SQL_CON)
    total_rows = stats['nrows'][0]

    sample_memory_usage = sample.memory_usage(deep=True).sum()
    sample_rows = len(sample)

    est_memory_size = (total_rows / sample_rows) * sample_memory_usage

    return est_memory_size


def _free_memory():
    return SAFETY_FACTOR * psutil.virtual_memory().available
