import unittest
import pandas as pd
import sqlite3
from core import Table, Projection, Selection, Join, SQL_CON, \
    _get_dependency_graph, _topological_sort


class TestPandaSQL(unittest.TestCase):
    # TODO: refactor into multiple unit tests when there is enough
    # functionality to test

    def test_simple_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = Table(base_df)
        proj = df['n']
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n FROM {}'.format(df.name))

        result = proj.compute()
        expected = base_df[['n']]
        pd.testing.assert_frame_equal(result, expected)

    def test_multiple_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = Table(base_df)
        proj = df[['n', 's']]
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n, s FROM {}'.format(df.name))

        result = proj.compute()
        expected = base_df[['n', 's']]
        pd.testing.assert_frame_equal(result, expected)

    def test_simple_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = Table(base_df)
        selection = df[df['n'] == 5]
        self.assertIsInstance(selection, Selection)

        sql = selection.sql()
        self.assertEqual(sql, 'SELECT * FROM {} WHERE {}.n = 5'
                         .format(df.name, df.name))

        result = selection.compute()
        expected = base_df[base_df['n'] == 5]

        # Ignore Pandas index for the comparison
        result.reset_index(drop=True, inplace=True)
        expected.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_nested_operation(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = Table(base_df)
        selection = df[df['n'] == 5]
        selection.name = 'S'
        proj = selection['s']

        sql = proj.sql()
        self.assertEqual(sql, 'WITH {} AS (SELECT * FROM {} WHERE {}.n = 5) '
                         'SELECT s FROM {}'.format(selection.name, df.name,
                                                   df.name, selection.name))

        result = proj.compute()
        expected = base_df[base_df['n'] == 5][['s']]

        # Ignore Pandas index for the comparison
        result.reset_index(drop=True, inplace=True)
        expected.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_simple_join(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = Table(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = Table(base_df_2)
        joined = df_1.join(df_2, on='n')
        self.assertIsInstance(joined, Join)

        sql = joined.sql()
        self.assertEqual(sql, 'SELECT * FROM {} JOIN {} USING (n)'
                         .format(df_1.name, df_2.name))

        result = joined.compute()
        expected = base_df_1.merge(base_df_2, on='n')

        # Ignore Pandas index for the comparison
        result.reset_index(drop=True, inplace=True)
        expected.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(result, expected)

    def test_get_dependency_graph(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = Table(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = Table(base_df_2)
        joined = df_1.join(df_2, on='n')

        graph = _get_dependency_graph(joined)
        self.assertIn(df_1, graph)
        self.assertIn(df_2, graph)
        self.assertIn(joined, graph)
        self.assertIn(df_1, set(graph[joined]))
        self.assertIn(df_2, set(graph[joined]))
        self.assertEqual(len(graph[joined]), 2)
        self.assertEqual(len(graph[df_1]), 0)
        self.assertEqual(len(graph[df_2]), 0)

    def test_topological_sort(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = Table(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = Table(base_df_2)
        joined = df_1.join(df_2, on='n')

        graph = _get_dependency_graph(joined)
        ordered = _topological_sort(graph)
        self.assertEqual(ordered[0].name, df_1.name)
        self.assertEqual(ordered[1].name, df_2.name)
        self.assertEqual(ordered[2].name, joined.name)

    def test_join_with_dependencies(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        T1 = Table(base_df_1)
        T2 = Table(base_df_2)
        S1 = T1[T1['n'] < 8]
        S2 = T2[T2['n'] >= 3]
        joined = S1.join(S2, on='n')

        self.assertIsInstance(joined, Join)
        self.assertEqual(joined.sql(), 'WITH {} AS ({}), {} AS ({}) '
                         'SELECT * FROM {} JOIN {} USING (n)'
                         .format(S1.name, S1.sql(False),
                                 S2.name, S2.sql(False),
                                 S1.name, S2.name))

        result = joined.compute()
        expected = base_df_1[base_df_1['n'] < 8].merge(
            base_df_2[base_df_2['n'] >= 3], on='n')

        # Ignore Pandas index for the comparison
        result.reset_index(drop=True, inplace=True)
        expected.reset_index(drop=True, inplace=True)
        pd.testing.assert_frame_equal(result, expected)


if __name__ == "__main__":
    unittest.main()
