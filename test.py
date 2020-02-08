import unittest
import pandas as pd
import sqlite3
from core import BaseTable, Projection, Selection, Join, SQL_CON


class TestPandaSQL(unittest.TestCase):
    # TODO: refactor into multiple unit tests when there is enough
    # functionality to test

    def test_from_pandas(self):
        df = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val': str(i*2)} for i in range(10)]))
        self.assertIsInstance(df, BaseTable)
        self.assertIsInstance(SQL_CON, sqlite3.Connection)

    def test_simple_projection(self):
        df = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val': str(i*2)} for i in range(10)]))
        df.name = 'T'
        proj = df['num']
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT num FROM T')

    def test_multiple_projection(self):
        df = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val': str(i*2)} for i in range(10)]))
        df.name = 'T'
        proj = df[['num', 'val']]
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT num, val FROM T')

    def test_simple_selection(self):
        df = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val': str(i*2)} for i in range(10)]))
        df.name = 'T'
        selection = df[df['num'] == 10]
        self.assertIsInstance(selection, Selection)

        sql = selection.sql()
        self.assertEqual(sql, 'SELECT * FROM T WHERE T.num = 10')

    def test_nested_operation(self):
        df = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val': str(i*2)} for i in range(10)]))
        df.name = 'T'
        selection = df[df['num'] == 10]
        selection.name = 'S'
        proj = selection['val']

        sql = proj.sql()
        self.assertEqual(sql, 'WITH S AS (SELECT * FROM T WHERE T.num = 10) '
                         'SELECT val FROM S')

    def test_simple_join(self):
        df1 = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val1': str(i*2)} for i in range(10)]))
        df1.name = 'S'
        df2 = BaseTable.from_pandas(pd.DataFrame(
            [{'num': i, 'val2': str(i*2)} for i in range(10)]))
        df2.name = 'T'
        joined = df1.join(df2, on='num')
        self.assertIsInstance(joined, Join)

        sql = joined.sql()
        self.assertEqual(sql, 'SELECT * FROM S JOIN T ON S.num = T.num')


if __name__ == "__main__":
    unittest.main()
