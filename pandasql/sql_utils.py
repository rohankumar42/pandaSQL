import sqlite3


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


def get_sqlite_connection():
    con = sqlite3.connect(":memory:")

    for name, (func, num_args) in CUSTOM_FUNCTIONS.items():
        con.create_function(name, num_args, func)

    return con
