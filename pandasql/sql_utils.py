import duckdb
import sqlite3


##############################################################################
#                           Custom Arithmetic
##############################################################################


def mod(x: int, y: int):
    """Mod with Python/Pandas semantics instead of the SQLite's %"""
    return x % y


def div(x, y):
    """Truediv with Python/Pandas semantics instead of the SQLite's /"""
    return x / y


def floor_div(x, y):
    return x // y


def inv(x):
    return ~x


def bit_and(x, y):
    return x & y


def bit_or(x, y):
    return x | y


def bit_xor(x, y):
    return x ^ y


CUSTOM_FUNCTIONS = {
    "POW": (pow, 2),
    "MOD": (mod, 2),
    "DIV": (div, 2),
    "INV": (inv, 1),
    "FLOORDIV": (floor_div, 2),
    "BITAND": (bit_and, 2),
    "BITOR": (bit_or, 2),
    "BITXOR": (bit_xor, 2),
}


##############################################################################
#                           Custom Aggregators
##############################################################################


class Fold:
    def __init__(self, func, base):
        self.result = base
        self.func = func

    def step(self, item):
        self.result = self.func(self.result, item)

    def finalize(self):
        return self.result


class Prod(Fold):
    def __init__(self):
        super().__init__(func=lambda x, y: x * y, base=1)


class Any(Fold):
    def __init__(self):
        super().__init__(func=lambda x, y: x or y, base=False)


class All(Fold):
    def __init__(self):
        super().__init__(func=lambda x, y: x and y, base=True)


CUSTOM_AGGREGATORS = {
    "PROD": (Prod, 1),
    "AGG_ANY": (Any, 1),
    "AGG_ALL": (All, 1),
}


##############################################################################
#                           SQL Connection Utils
##############################################################################

def get_sqlite_connection(file_name):
    # TODO: Figure out if it's possible to delete this file when the Python
    # process terminates

    con = sqlite3.connect(file_name)

    for name, (func, num_args) in CUSTOM_FUNCTIONS.items():
        con.create_function(name, num_args, func)

    for name, (cls, num_args) in CUSTOM_AGGREGATORS.items():
        con.create_aggregate(name, num_args, cls)

    c = con.cursor()
    c.execute(''' PRAGMA compile_options; ''')
    options = c.fetchall()

    dbstat_enabled = [o for o in options if 'ENABLE_DBSTAT_VTAB' in o]

    if len(dbstat_enabled) == 0:
        raise ImportError('sqlite3 must be compiled with '
                          'SQLITE_ENABLE_DBSTAT_VTAB to use pandaSQL')
    return con


def get_duckdb_connection(file_name):
    print('Starting duckdb connection to file', file_name)
    con = duckdb.connect(file_name)
    return con
