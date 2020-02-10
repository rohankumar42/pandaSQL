import unittest
import pandas as pd
import sqlite3
from core import Table, Projection, Selection, Join, SQL_CON, \
    _get_dependency_graph, _topological_sort


class TestPandaSQL(unittest.TestCase):
    # TODO: refactor into multiple unit tests when there is enough
    # functionality to test

    def test_load(self):
        df = Table(pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)]))
        self.assertIsInstance(df, Table)
        self.assertIsInstance(SQL_CON, sqlite3.Connection)

    def test_simple_projection(self):
        df = Table(pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)]))
        df.name = 'T'
        proj = df['n']
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n FROM T')

    def test_multiple_projection(self):
        df = Table(pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)]))
        df.name = 'T'
        proj = df[['n', 's']]
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n, s FROM T')

    def test_simple_selection(self):
        df = Table(pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)]))
        df.name = 'T'
        selection = df[df['n'] == 10]
        self.assertIsInstance(selection, Selection)

        sql = selection.sql()
        self.assertEqual(sql, 'SELECT * FROM T WHERE T.n = 10')

    def test_nested_operation(self):
        df = Table(pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)]))
        df.name = 'T'
        selection = df[df['n'] == 10]
        selection.name = 'S'
        proj = selection['s']

        sql = proj.sql()
        self.assertEqual(sql, 'WITH S AS (SELECT * FROM T WHERE T.n = 10) '
                         'SELECT s FROM S')

    def test_simple_join(self):
        df1 = Table(pd.DataFrame(
            [{'n': i, 's1': str(i*2)} for i in range(10)]))
        df1.name = 'S'
        df2 = Table(pd.DataFrame(
            [{'n': i, 's2': str(i*2)} for i in range(10)]))
        df2.name = 'T'
        joined = df1.join(df2, on='n')
        self.assertIsInstance(joined, Join)

        sql = joined.sql()
        self.assertEqual(sql, 'SELECT * FROM S JOIN T ON S.n = T.n')

    def test_get_dependency_graph(self):
        df1 = Table(pd.DataFrame(
            [{'n': i, 's1': str(i*2)} for i in range(10)]))
        df1.name = 'S'
        df2 = Table(pd.DataFrame(
            [{'n': i, 's2': str(i*2)} for i in range(10)]))
        df2.name = 'T'
        joined = df1.join(df2, on='n')

        graph = _get_dependency_graph(joined)
        self.assertIn(df1, graph)
        self.assertIn(df2, graph)
        self.assertIn(joined, graph)
        self.assertIn(df1, set(graph[joined]))
        self.assertIn(df2, set(graph[joined]))
        self.assertEqual(len(graph[joined]), 2)
        self.assertEqual(len(graph[df1]), 0)
        self.assertEqual(len(graph[df2]), 0)

    def test_topological_sort(self):
        df1 = Table(pd.DataFrame(
            [{'n': i, 's1': str(i*2)} for i in range(10)]))
        df1.name = 'S'
        df2 = Table(pd.DataFrame(
            [{'n': i, 's2': str(i*2)} for i in range(10)]))
        df2.name = 'T'
        joined = df1.join(df2, on='n')

        graph = _get_dependency_graph(joined)
        ordered = _topological_sort(graph)
        self.assertEqual(ordered[0].name, 'S')
        self.assertEqual(ordered[1].name, 'T')
        self.assertEqual(ordered[2].name, joined.name)

    def test_join_with_dependencies(self):
        df = pd.DataFrame([{'n': i, 's': str(i)} for i in range(10)])
        T1 = Table(df)
        T2 = Table(df)
        S1 = T1[T1['n'] < 5]
        S1.name = 'S1'
        S2 = T2[T2['n'] >= 5]
        S2.name = 'S2'
        joined = S1.join(S2, on='n')

        self.assertIsInstance(joined, Join)
        self.assertEqual(joined.sql(), 'WITH S1 AS ({}), S2 AS ({}) '
                         'SELECT * FROM S1 JOIN S2 ON S1.n = S2.n'
                         .format(S1.sql(False), S2.sql(False)))


if __name__ == "__main__":
    unittest.main()
