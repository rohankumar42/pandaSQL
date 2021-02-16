import os
from .core import DataFrame, stop, offloading_strategy, use_memory_prediction, \
    concat, merge, get_database_file, set_database_file  # noqa
from .io import read_csv, read_json, read_numpy, read_pickle  # noqa

__all__ = [
    'DataFrame', 'concat', 'merge',
    'stop', 'offloading_strategy', 'get_database_file', 'set_database_file',
    'read_csv', 'read_json', 'read_numpy', 'read_pickle'
]

offloading_strategy(os.getenv('PANDASQL_OFFLOADING') or 'ALWAYS')
use_memory_prediction(bool(os.getenv('PANDASQL_MEMORY_PREDICTION') or 'True'))
