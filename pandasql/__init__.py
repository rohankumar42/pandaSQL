import os
from .core import DataFrame, stop, offloading_strategy, concat, merge  # noqa
from .io import read_csv, read_json, read_numpy, read_pickle  # noqa

__all__ = [
    'DataFrame', 'stop', 'offloading_strategy'
    'concat', 'merge',
    'read_csv', 'read_json', 'read_numpy', 'read_pickle'
]

offloading_strategy(os.getenv('PANDASQL_OFFLOADING') or 'ALWAYS')
