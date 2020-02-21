import unittest
import pandas as pd
import pandasql as ps


class TestDataFrame(unittest.TestCase):
    # TODO: refactor into multiple unit tests when there is enough
    # functionality to test

    def assertDataFrameEqualsPandas(self, df: ps.DataFrame,
                                    expected_df: pd.DataFrame):
        result = df.compute()

        # Ignore Pandas index for the comparison
        result.reset_index(drop=True, inplace=True)
        expected_df.reset_index(drop=True, inplace=True)

        pd.testing.assert_frame_equal(result, expected_df)

    def test_simple_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        proj = df['n']
        self.assertIsInstance(proj, ps.Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n FROM {}'.format(df.name))

        expected = base_df[['n']]
        self.assertDataFrameEqualsPandas(proj, expected)

    def test_multiple_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        proj = df[['n', 's']]
        self.assertIsInstance(proj, ps.Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n, s FROM {}'.format(df.name))

        expected = base_df[['n', 's']]
        self.assertDataFrameEqualsPandas(proj, expected)

    def test_simple_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        selection = df[df['n'] == 5]
        self.assertIsInstance(selection, ps.Selection)

        sql = selection.sql()
        self.assertEqual(sql, 'SELECT * FROM {} WHERE {}.n = 5'
                         .format(df.name, df.name))

        expected = base_df[base_df['n'] == 5]
        self.assertDataFrameEqualsPandas(selection, expected)

    def test_nested_operation(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        selection = df[df['n'] == 5]
        proj = selection['s']

        sql = proj.sql()
        self.assertEqual(sql, 'WITH {} AS (SELECT * FROM {} WHERE {}.n = 5) '
                         'SELECT s FROM {}'.format(selection.name, df.name,
                                                   df.name, selection.name))

        expected = base_df[base_df['n'] == 5][['s']]
        self.assertDataFrameEqualsPandas(proj, expected)

    def test_limit_after_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        selection = df[df['n'] != 0]
        limit = selection[:5]

        sql = limit.sql()
        self.assertEqual(sql, 'WITH {} AS (SELECT * FROM {} WHERE {}.n <> 0) '
                         'SELECT * FROM {} LIMIT 5'
                         .format(selection.name, df.name,
                                 df.name, selection.name))

        expected = base_df[base_df['n'] != 0][:5]
        self.assertDataFrameEqualsPandas(limit, expected)

        no_limit = selection[:]

        sql = no_limit.sql()
        self.assertEqual(sql, 'SELECT * FROM {} WHERE {}.n <> 0'
                         .format(df.name, df.name))

        expected = base_df[base_df['n'] != 0][:]
        self.assertDataFrameEqualsPandas(no_limit, expected)

    def test_order_by(self):
        base_df = pd.DataFrame([{'x': i // 2, 'y': i % 2} for i in range(10)])
        df = ps.DataFrame(base_df)

        # Sort on one column
        ordered = df.sort_values('x', ascending=False)
        sql = ordered.sql()
        self.assertEqual(sql, 'SELECT * FROM {} ORDER BY {}.x DESC'
                         .format(df.name, df.name))

        expected = base_df.sort_values('x', ascending=False)
        self.assertDataFrameEqualsPandas(ordered, expected)

        # Sort on multiple columns
        ordered = df.sort_values(['x', 'y'], ascending=[True, False])
        sql = ordered.sql()
        self.assertEqual(sql, 'SELECT * FROM {} ORDER BY {}.x ASC, {}.y DESC'
                         .format(df.name, df.name, df.name))

        expected = base_df.sort_values(['x', 'y'], ascending=[True, False])
        self.assertDataFrameEqualsPandas(ordered, expected)

    def test_simple_join(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = ps.DataFrame(base_df_2)
        joined = df_1.join(df_2, on='n')
        self.assertIsInstance(joined, ps.Join)

        sql = joined.sql()
        self.assertEqual(sql, 'SELECT * FROM {} JOIN {} USING (n)'
                         .format(df_1.name, df_2.name))

        expected = base_df_1.merge(base_df_2, on='n')
        self.assertDataFrameEqualsPandas(joined, expected)

    def test_join_with_dependencies(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        T1 = ps.DataFrame(base_df_1)
        T2 = ps.DataFrame(base_df_2)
        S1 = T1[T1['n'] < 8]
        S2 = T2[T2['n'] >= 3]
        joined = S1.join(S2, on='n')

        self.assertIsInstance(joined, ps.Join)
        self.assertEqual(joined.sql(), 'WITH {} AS ({}), {} AS ({}) '
                         'SELECT * FROM {} JOIN {} USING (n)'
                         .format(S1.name, S1.sql(False),
                                 S2.name, S2.sql(False),
                                 S1.name, S2.name))

        expected = base_df_1[base_df_1['n'] < 8].merge(
            base_df_2[base_df_2['n'] >= 3], on='n')
        self.assertDataFrameEqualsPandas(joined, expected)

    def test_complex_criteria(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        T = ps.DataFrame(base_df)
        conj = T[(T['n'] > 2) & (T['n'] <= 6)]
        disj = T[(T['n'] < 2) | (T['n'] >= 6)]
        neg = T[~(T['s'] != '2')]

        self.assertEqual(conj.sql(), 'SELECT * FROM {} WHERE '
                         '{}.n > 2 AND {}.n <= 6'
                         .format(T.name, T.name, T.name))
        self.assertEqual(disj.sql(), 'SELECT * FROM {} WHERE '
                         '{}.n < 2 OR {}.n >= 6'
                         .format(T.name, T.name, T.name))
        self.assertEqual(neg.sql(), 'SELECT * FROM {} WHERE NOT ({}.s <> 2)'
                         .format(T.name, T.name))

        conj_expected = base_df[(base_df['n'] > 2) & (base_df['n'] <= 6)]
        self.assertDataFrameEqualsPandas(conj, conj_expected)

        disj_expected = base_df[(base_df['n'] < 2) | (base_df['n'] >= 6)]
        self.assertDataFrameEqualsPandas(disj, disj_expected)

        neg_expected = base_df[~(base_df['s'] != '2')]
        self.assertDataFrameEqualsPandas(neg, neg_expected)

    def test_implicit_compute_on_len(self):
        T = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        selection = T[T['n'] < 5]
        self.assertIsNone(selection.result)
        n = len(selection)
        self.assertIsInstance(selection.result, pd.DataFrame)
        self.assertEqual(n, 5)

    def test_implicit_compute_on_str(self):
        df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        T = ps.DataFrame(df)
        selection = T[T['n'] < 5]
        self.assertIsNone(selection.result)
        out = str(selection)
        self.assertIsInstance(selection.result, pd.DataFrame)
        self.assertEqual(out, str(df[df['n'] < 5]))

    def test_caching_in_sqlite(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        T1 = ps.DataFrame(base_df_1)
        T2 = ps.DataFrame(base_df_2)
        S1 = T1[T1['n'] < 8]
        S2 = T2[T2['n'] >= 3]
        joined = S1.join(S2, on='n')

        # Compute S1, so its result is cached in SQLite
        S1.compute()

        # Now, the query for joined should not declare S1 again
        self.assertEqual(joined.sql(), 'WITH {} AS ({}) '
                         'SELECT * FROM {} JOIN {} USING (n)'
                         .format(S2.name, S2.sql(False),
                                 S1.name, S2.name))

        expected = base_df_1[base_df_1['n'] < 8].merge(
            base_df_2[base_df_2['n'] >= 3], on='n')
        self.assertDataFrameEqualsPandas(joined, expected)


if __name__ == "__main__":
    unittest.main()
