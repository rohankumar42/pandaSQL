import os
from .core import DataFrame, stop, offloading_strategy, concat, merge, \
    get_database_file, set_database_file  # noqa
from .io import read_csv, read_json, read_numpy, read_pickle  # noqa

__all__ = [
    'DataFrame', 'concat', 'merge',
    'stop', 'offloading_strategy', 'get_database_file', 'set_database_file',
    'read_csv', 'read_json', 'read_numpy', 'read_pickle'
]

offloading_strategy(os.getenv('PANDASQL_OFFLOADING') or 'ALWAYS')
