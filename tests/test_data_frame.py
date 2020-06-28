import unittest

import pandas as pd
import pandasql as ps
from pandasql.core import Selection, Projection, Union, Join, Limit, OrderBy, \
    Aggregator, GroupByDataFrame, GroupByProjection
from .utils import assertDataFrameEqualsPandas


class TestDataFrame(unittest.TestCase):
    # TODO: refactor into multiple unit tests when there is enough
    # functionality to test

    def setUp(self):
        ps.offloading_strategy('ALWAYS')

    def test_simple_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        proj = df['n']
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n FROM {}'.format(df.name))

        expected = base_df[['n']]
        assertDataFrameEqualsPandas(proj, expected)

    def test_multiple_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        proj = df[['n', 's']]
        self.assertIsInstance(proj, Projection)

        sql = proj.sql()
        self.assertEqual(sql, 'SELECT n, s FROM {}'.format(df.name))

        expected = base_df[['n', 's']]
        assertDataFrameEqualsPandas(proj, expected)

    def test_simple_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        selection = df[df['n'] == 5]
        self.assertIsInstance(selection, Selection)

        sql = selection.sql()
        self.assertEqual(sql, 'SELECT * FROM {} WHERE {}.n = 5'
                         .format(df.name, df.name))

        expected = base_df[base_df['n'] == 5]
        assertDataFrameEqualsPandas(selection, expected)

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
        assertDataFrameEqualsPandas(proj, expected)

    def test_limit_after_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        selection = df[df['n'] != 0]
        limit = selection[:5]
        self.assertIsInstance(limit, Limit)

        sql = limit.sql()
        self.assertEqual(sql, 'WITH {} AS (SELECT * FROM {} WHERE {}.n <> 0) '
                         'SELECT * FROM {} LIMIT 5'
                         .format(selection.name, df.name,
                                 df.name, selection.name))

        expected = base_df[base_df['n'] != 0][:5]
        assertDataFrameEqualsPandas(limit, expected)

        no_limit = selection[:]

        sql = no_limit.sql()
        self.assertEqual(sql, 'SELECT * FROM {} WHERE {}.n <> 0'
                         .format(df.name, df.name))

        expected = base_df[base_df['n'] != 0][:]
        assertDataFrameEqualsPandas(no_limit, expected)

    def test_order_by(self):
        base_df = pd.DataFrame([{'x': i // 2, 'y': i % 2} for i in range(10)])
        df = ps.DataFrame(base_df)

        # Sort on one column
        ordered = df.sort_values('x', ascending=False)
        self.assertIsInstance(ordered, OrderBy)

        sql = ordered.sql()
        self.assertEqual(sql, 'SELECT * FROM {} ORDER BY {}.x DESC'
                         .format(df.name, df.name))

        expected = base_df.sort_values('x', ascending=False)
        assertDataFrameEqualsPandas(ordered, expected)

        # Sort on multiple columns
        ordered = df.sort_values(['x', 'y'], ascending=[True, False])
        sql = ordered.sql()
        self.assertEqual(sql, 'SELECT * FROM {} ORDER BY {}.x ASC, {}.y DESC'
                         .format(df.name, df.name, df.name))

        expected = base_df.sort_values(['x', 'y'], ascending=[True, False])
        assertDataFrameEqualsPandas(ordered, expected)

    def test_simple_union(self):
        base_df_1 = pd.DataFrame([{'n': i, 's': str(i)} for i in range(8)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's': str(i)} for i in range(4, 12)])
        df_2 = ps.DataFrame(base_df_2)
        base_df_3 = pd.DataFrame([{'n': i, 's': str(i)} for i in range(8, 16)])
        df_3 = ps.DataFrame(base_df_3)

        union = ps.concat([df_1, df_2, df_3])
        self.assertIsInstance(union, Union)

        sql = union.sql()
        self.assertEqual(sql, 'SELECT * FROM {} UNION ALL SELECT * FROM {} '
                         'UNION ALL SELECT * FROM {}'
                         .format(df_1.name, df_2.name, df_3.name))

        expected = pd.concat([base_df_1, base_df_2, base_df_3])
        pd.testing.assert_index_equal(union.columns, expected.columns)
        assertDataFrameEqualsPandas(union, expected)

    def test_simple_merge(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = ps.DataFrame(base_df_2)

        merged = df_1.merge(df_2, on='n')
        self.assertIsInstance(merged, Join)

        sql = merged.sql()
        self.assertEqual(sql,
                         'SELECT {}.n AS n, {}.s1 AS s1, {}.s2 AS s2 FROM '
                         '{} JOIN {} ON ({}.n = {}.n)'
                         .format(df_1.name, df_1.name, df_2.name, df_1.name,
                                 df_2.name, df_1.name, df_2.name))

        expected = base_df_1.merge(base_df_2, on='n')
        assertDataFrameEqualsPandas(merged, expected)

    def test_simple_merge_on_different_columns(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'m': i, 's2': str(i*2)} for i in range(10)])
        df_2 = ps.DataFrame(base_df_2)

        merged = df_1.merge(df_2, left_on='n', right_on='m')
        self.assertIsInstance(merged, Join)

        sql = merged.sql()
        self.assertEqual(sql,
                         'SELECT {}.m AS m, {}.n AS n, {}.s1 AS s1, {}.s2 AS s2'  # noqa
                         ' FROM {} JOIN {} ON ({}.n = {}.m)'
                         .format(df_2.name, df_1.name, df_1.name, df_2.name,
                                 df_1.name, df_2.name, df_1.name, df_2.name))

        expected = base_df_1.merge(base_df_2, left_on='n', right_on='m')
        assertDataFrameEqualsPandas(merged, expected)

    def test_merge_with_dependencies(self):
        self.maxDiff = None
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        T1 = ps.DataFrame(base_df_1)
        T2 = ps.DataFrame(base_df_2)
        S1 = T1[T1['n'] < 8]
        S2 = T2[T2['n'] >= 3]
        merged = S1.merge(S2, on='n')

        self.assertIsInstance(merged, Join)
        self.assertEqual(merged.sql(), 'WITH {} AS ({}), {} AS ({}) '
                         'SELECT {}.n AS n, {}.s1 AS s1, {}.s2 AS s2 FROM '
                         '{} JOIN {} ON ({}.n = {}.n)'
                         .format(S1.name, S1.sql(False),
                                 S2.name, S2.sql(False),
                                 S1.name, S1.name, S2.name,
                                 S1.name, S2.name, S1.name, S2.name))

        expected = base_df_1[base_df_1['n'] < 8].merge(
            base_df_2[base_df_2['n'] >= 3], on='n')
        assertDataFrameEqualsPandas(merged, expected)

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
        self.assertEqual(neg.sql(), "SELECT * FROM {} WHERE NOT ({}.s <> '2')"
                         .format(T.name, T.name))

        conj_expected = base_df[(base_df['n'] > 2) & (base_df['n'] <= 6)]
        assertDataFrameEqualsPandas(conj, conj_expected)

        disj_expected = base_df[(base_df['n'] < 2) | (base_df['n'] >= 6)]
        assertDataFrameEqualsPandas(disj, disj_expected)

        neg_expected = base_df[~(base_df['s'] != '2')]
        assertDataFrameEqualsPandas(neg, neg_expected)

    def test_columns(self):
        df = pd.DataFrame([{'a': i, 'b': i*2, 'c': i-1} for i in range(10)])
        T = ps.DataFrame(df)
        out = T[T['a'] < 5]

        pd.testing.assert_index_equal(out[['c']].columns, df[['c']].columns)
        pd.testing.assert_index_equal(out[['b', 'c']].columns,
                                      df[['b', 'c']].columns)

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
        merged = S1.merge(S2, on='n')

        # Compute S1, so its result is cached in SQLite
        S1.compute()

        # Now, the query for merged should not declare S1 again
        self.assertEqual(merged.sql(), 'WITH {} AS ({}) '
                         'SELECT {}.n AS n, {}.s1 AS s1, {}.s2 AS s2 FROM '
                         '{} JOIN {} ON ({}.n = {}.n)'
                         .format(S2.name, S2.sql(False),
                                 S1.name, S1.name, S2.name,
                                 S1.name, S2.name, S1.name, S2.name))

        expected = base_df_1[base_df_1['n'] < 8].merge(
            base_df_2[base_df_2['n'] >= 3], on='n')
        assertDataFrameEqualsPandas(merged, expected)

    def test_duplicating_column(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        # New column
        df['b'] = df['a']
        base_df['b'] = base_df['a']

        pd.testing.assert_index_equal(df.columns, base_df.columns)
        assertDataFrameEqualsPandas(df, base_df)

    def test_write_constant_column(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        # New columns
        df['b'] = 10
        df['c'] = 'dummy'

        # Same operations on Pandas
        base_df['b'] = 10
        base_df['c'] = 'dummy'

        pd.testing.assert_index_equal(df.columns, base_df.columns)
        assertDataFrameEqualsPandas(df, base_df)

    def test_write_on_downstream_dataframe(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        # New columns
        selection = df[df['a'] != '4']
        selection['b'] = 10

        # Same operations on Pandas
        expected = base_df[base_df['a'] != '4']
        # Make new copy to avoid Pandas warning about writing to a slice
        expected = pd.DataFrame(expected)
        expected['b'] = 10

        pd.testing.assert_index_equal(selection.columns, expected.columns)
        assertDataFrameEqualsPandas(selection, expected)

    def test_old_dependents_after_write(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        old_proj = df['a']
        expected_old_proj = base_df[['a']]

        # Change values in column
        df['a'] = df['n']
        base_df['a'] = base_df['n']

        # df should have the updated column
        pd.testing.assert_index_equal(df.columns, base_df.columns)
        assertDataFrameEqualsPandas(df, base_df)

        # But old_proj should have old values of column
        assertDataFrameEqualsPandas(old_proj, expected_old_proj)

    def test_simple_arithmetic_on_columns(self):
        base_df = pd.DataFrame([{'n': i, 'm': 10-i} for i in range(10)])
        df = ps.DataFrame(base_df)

        res = df['n'] + 2 * df['m']
        base_res = base_df['n'] + 2 * base_df['m']

        self.assertEqual(res.sql(),
                         'SELECT {}.n + (2 * {}.m) AS res FROM {}'
                         .format(df.name, df.name, df.name))

        expected = pd.DataFrame()
        expected['res'] = base_res

        assertDataFrameEqualsPandas(res, expected)

    def test_complex_arithmetic_on_columns(self):
        base_df = pd.DataFrame([{'n': i, 'm': 10-i} for i in range(1, 11)])
        df = ps.DataFrame(base_df)

        res = 3 / ((abs(-df['n'] // 2) ** df['m']) % 13)
        base_res = 3 / ((abs(-base_df['n'] // 2) ** base_df['m']) % 13)

        self.assertEqual(res.sql(),
                         'SELECT DIV(3, MOD(POW(abs(FLOORDIV(({}.n * -1), 2)), {}.m), 13)) AS res FROM {}'  # noqa
                         .format(df.name, df.name, df.name))

        expected = pd.DataFrame()
        expected['res'] = base_res

        assertDataFrameEqualsPandas(res, expected)

    def test_bitwise_operations_on_columns(self):
        base_df = pd.DataFrame([{'n': i, 'm': 10-i} for i in range(1, 11)])
        df = ps.DataFrame(base_df)

        res = (~df['n'] & (df['m'] % 2)) ^ (2 | df['m'])
        base_res = (~base_df['n'] & (base_df['m'] % 2)) ^ (2 | base_df['m'])

        self.assertEqual(res.sql(),
                         'SELECT BITXOR(BITAND(INV({}.n), MOD({}.m, 2)), BITOR(2, {}.m)) AS res FROM {}'  # noqa
                         .format(df.name, df.name, df.name, df.name))

        expected = pd.DataFrame()
        expected['res'] = base_res

        assertDataFrameEqualsPandas(res, expected)

    def test_column_sums(self):
        base_df = pd.DataFrame([{'m': i, 'n': 10-i} for i in range(1, 11)])
        df = ps.DataFrame(base_df)

        res = df.sum()
        base_res = base_df.sum()

        self.assertIsInstance(res, Aggregator)
        self.assertEqual(res.sql(), 'SELECT SUM(m) AS m, SUM(n) AS n FROM {}'
                         .format(df.name))

        # Compare Pandas series
        assertDataFrameEqualsPandas(res, base_res.astype(int))

        res = df['n'].sum()
        base_res = base_df['n'].sum()

        # Compare ints
        self.assertEqual(res.compute(), base_res)

    def test_aggregation_operations(self):
        base_df = pd.DataFrame([{'m': i, 'n': 10-i} for i in range(0, 10)])
        df = ps.DataFrame(base_df)

        assertDataFrameEqualsPandas(df.sum(), base_df.sum())
        assertDataFrameEqualsPandas(df.mean(), base_df.mean())
        assertDataFrameEqualsPandas(df.count(), base_df.count())
        assertDataFrameEqualsPandas(df.min(), base_df.min())
        assertDataFrameEqualsPandas(df.max(), base_df.max())
        assertDataFrameEqualsPandas(df.prod(), base_df.prod())
        assertDataFrameEqualsPandas(df.any(), base_df.any())
        assertDataFrameEqualsPandas(df.all(), base_df.all())

    def test_simple_groupby_sum(self):
        base_df = pd.DataFrame([
            {'a': str(i), 'b': str(j), 'c': 100*i, 'd': -j}
            for i in range(3) for j in range(3)
        ])
        df = ps.DataFrame(base_df)

        grouped = df.groupby(['a', 'b'])
        base_grouped = base_df.groupby(['a', 'b'], as_index=False)

        res = grouped.sum()
        base_res = base_grouped.sum()

        self.assertIsInstance(grouped, GroupByDataFrame)
        self.assertIsInstance(res, Aggregator)
        self.assertEqual(res.sql(), 'SELECT a, b, SUM(c) AS c, SUM(d) AS d '
                         'FROM {} GROUP BY a, b'.format(df.name))

        base_res = pd.DataFrame().append(base_res, ignore_index=True)
        assertDataFrameEqualsPandas(res, base_res)

    def test_groupby_projection_sum(self):
        base_df = pd.DataFrame([
            {'a': str(i), 'b': str(j), 'c': 100*i, 'd': -j}
            for i in range(3) for j in range(3)
        ])
        df = ps.DataFrame(base_df)

        grouped = df.groupby(['a', 'b'])['c']
        base_grouped = base_df.groupby(['a', 'b'], as_index=False)['c']

        res = grouped.sum()
        base_res = base_grouped.sum()

        self.assertIsInstance(grouped, GroupByProjection)
        self.assertIsInstance(res, Aggregator)
        self.assertEqual(res.sql(), 'SELECT a, b, SUM(c) AS c '
                         'FROM {} GROUP BY a, b'.format(df.name))

        base_res = pd.DataFrame().append(base_res, ignore_index=True)
        assertDataFrameEqualsPandas(res, base_res)

    def test_groupby_further_usage(self):
        base_df = pd.DataFrame([{'r': i // 3, 'n': i, 'm': 2*i}
                                for i in range(1, 9)])
        df = ps.DataFrame(base_df)

        # TODO: as_index=True does not work because indexes are not
        # currently synchronized with SQLite
        agg = df.groupby('r').sum()
        base_agg = base_df.groupby('r', as_index=False).sum()

        res = agg[agg['n'] > 10]
        base_res = base_agg[base_agg['n'] > 10]

        self.assertEqual(res.sql(), 'WITH {} AS ({}) '
                         'SELECT * FROM {} WHERE {}.n > 10'
                         .format(agg.name, agg.sql(), agg.name, agg.name))

        assertDataFrameEqualsPandas(res, base_res)

    def test_complex_read_query(self):
        base_df_1 = pd.DataFrame([
            {'a': str(i), 'b': str(j), 'c': 100*i, 'd': -j}
            for i in range(3) for j in range(3)
        ])
        base_df_2 = pd.DataFrame([
            {'a': str(i), 'b': str(j), 'e': 50*i, 'f': j}
            for i in range(3) for j in range(3)
        ])
        df_1 = ps.DataFrame(base_df_1)
        df_2 = ps.DataFrame(base_df_2)

        key = ['a', 'b']
        base_merged = base_df_1.merge(base_df_2, on=key)
        base_agg = base_merged.groupby(key, as_index=False)[['c', 'f']].sum()
        base_ordered = base_agg.sort_values(by=key, ascending=False)
        base_limit = base_ordered.head(3)

        merged = df_1.merge(df_2, on=key)
        agg = merged.groupby(key)[['c', 'f']].sum()
        ordered = agg.sort_values(by=key, ascending=False)
        limit = ordered.head(3)

        # All dependencies should have correct results
        assertDataFrameEqualsPandas(merged, base_merged)
        assertDataFrameEqualsPandas(agg, base_agg)
        assertDataFrameEqualsPandas(ordered, base_ordered)
        assertDataFrameEqualsPandas(limit, base_limit)

    def test_complex_write_query(self):
        base_df_1 = pd.DataFrame([
            {'a': i, 'b': j, 'c': 100*i, 'd': -j}
            for i in range(3) for j in range(3)
        ])
        base_df_2 = pd.DataFrame([
            {'a': i, 'b': j, 'e': 50*i, 'f': j}
            for i in range(3) for j in range(3)
        ])
        df_1 = ps.DataFrame(base_df_1)
        df_2 = ps.DataFrame(base_df_2)

        base_merged = base_df_1.merge(base_df_2, on=['a', 'b'])
        base_merged['diff'] = base_merged['c'] - base_merged['e']
        base_merged['key'] = base_merged['diff'] * \
            (base_merged['d'] - base_merged['f'])
        base_agg = base_merged.groupby('key', as_index=False)[['a', 'b']].sum()
        base_agg['sum'] = base_agg['a'] + base_agg['b']
        base_ordered = base_agg.sort_values(by='sum')

        merged = df_1.merge(df_2, on=['a', 'b'])
        merged['diff'] = merged['c'] - merged['e']
        merged['key'] = merged['diff'] * \
            (merged['d'] - merged['f'])
        agg = merged.groupby('key')[['a', 'b']].sum()
        agg['sum'] = agg['a'] + agg['b']
        ordered = agg.sort_values(by='sum')

        # All dependencies should have correct results
        assertDataFrameEqualsPandas(merged, base_merged)
        assertDataFrameEqualsPandas(agg, base_agg)
        assertDataFrameEqualsPandas(ordered, base_ordered)

    def test_rename_columns(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        base_nc = base_df.rename(columns={'n': 'b'})
        nc = df.rename(columns={'n': 'b'})

        base_nc['b'] += 1
        nc['b'] += 1
        assertDataFrameEqualsPandas(nc, base_nc)

    def test_rename_columns_and_compute(self):
        base_df = pd.DataFrame([{'n': i, 's': (i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        expected = base_df[base_df['n'].between(5, 9, inclusive=False)]
        expected = expected.rename(columns={'n': 'b'})

        df = df[df['n'] > 5]
        base_df = base_df[base_df['n'] > 5]

        base_nc = base_df.rename(columns={'n': 'b'})
        nc = df.rename(columns={'n': 'b'})

        nc = nc[nc['b'] < 9]
        base_nc = base_nc[base_nc['b'] < 9]


        assertDataFrameEqualsPandas(nc, base_nc)
        assertDataFrameEqualsPandas(nc, expected)


    def test_column_ordering(self):
        base_df = pd.DataFrame([{'n': i, 's': (i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_df['n'] += base_df['s']
        df['n'] += df['s']
        df.compute()

        self.assertListEqual(list(df.columns), list(base_df.columns))

if __name__ == "__main__":
    unittest.main()
