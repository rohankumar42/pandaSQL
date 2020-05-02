import unittest

import pandas as pd
import pandasql as ps
from pandasql.core import Selection, Projection, Union, Join, Limit, OrderBy, \
    Aggregator, GroupByDataFrame, GroupByProjection
from utils import assertDataFrameEqualsPandas


class TestPandasExecution(unittest.TestCase):

    def setUp(self):
        ps.offloading_strategy('NEVER')

    def test_projection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        assertDataFrameEqualsPandas(df['n'], base_df[['n']])
        assertDataFrameEqualsPandas(df[['n', 's']], base_df[['n', 's']])

    def test_criterion(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        assertDataFrameEqualsPandas(df['n'] == 5, base_df['n'] == 5)
        assertDataFrameEqualsPandas(df['n'] != 5, base_df['n'] != 5)
        assertDataFrameEqualsPandas(df['n'] >= 5, base_df['n'] >= 5)
        assertDataFrameEqualsPandas(df['n'] > 5, base_df['n'] > 5)
        assertDataFrameEqualsPandas(df['n'] <= 5, base_df['n'] <= 5)
        assertDataFrameEqualsPandas(~(df['n'] <= 5), ~(base_df['n'] <= 5))
        assertDataFrameEqualsPandas((df['n'] < 2) | (df['n'] > 6),
                                    (base_df['n'] < 2) | (base_df['n'] > 6))
        assertDataFrameEqualsPandas((df['n'] > 2) & (df['n'] < 6),
                                    (base_df['n'] > 2) & (base_df['n'] < 6))

    def test_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        assertDataFrameEqualsPandas(df[df['n'] == 5],
                                    base_df[base_df['n'] == 5])
        assertDataFrameEqualsPandas(df[(df['n'] < 2) | (df['n'] > 6)],
                                    base_df[(base_df['n'] < 2) | (base_df['n'] > 6)])  # noqa

    def test_projection_after_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        sel = df[df['n'] != 5]
        base_sel = base_df[base_df['n'] != 5]
        proj = sel['s']
        base_proj = base_sel[['s']]

        self.assertIsNone(sel.result)
        self.assertIsNone(proj.result)

        assertDataFrameEqualsPandas(proj, base_proj)

        # sel should also be cached because of the Pandas computation triggered
        self.assertIsNotNone(sel.result)
        self.assertIsNotNone(proj.result)

    def test_limit_after_selection(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        limit = df[df['n'] != 0][:5]
        head = df[df['n'] != 0].head(5)
        expected = base_df[base_df['n'] != 0].head()
        assertDataFrameEqualsPandas(limit, expected)
        assertDataFrameEqualsPandas(head, expected)

    def test_order_by(self):
        base_df = pd.DataFrame([{'x': i // 2, 'y': i % 2} for i in range(10)])
        df = ps.DataFrame(base_df)

        assertDataFrameEqualsPandas(df.sort_values('x', ascending=False),
                                    base_df.sort_values('x', ascending=False))

        assertDataFrameEqualsPandas(
            df.sort_values(['x', 'y'], ascending=[True, False]),
            base_df.sort_values(['x', 'y'], ascending=[True, False])
        )

    def test_union(self):
        base_df_1 = pd.DataFrame([{'n': i, 's': str(i)} for i in range(8)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's': str(i)} for i in range(4, 12)])
        df_2 = ps.DataFrame(base_df_2)
        base_df_3 = pd.DataFrame([{'n': i, 's': str(i)} for i in range(8, 16)])
        df_3 = ps.DataFrame(base_df_3)

        union = ps.concat([df_1, df_2, df_3])
        expected = pd.concat([base_df_1, base_df_2, base_df_3])
        assertDataFrameEqualsPandas(union, expected)

    def test_merge(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = ps.DataFrame(base_df_2)

        merged_a = df_1.merge(df_2, on='n')
        merged_b = ps.merge(df_1, df_2, on='n')
        expected = pd.merge(base_df_1, base_df_2, on='n')
        assertDataFrameEqualsPandas(merged_a, expected)
        assertDataFrameEqualsPandas(merged_b, expected)

    def test_arithmetic(self):
        base_df = pd.DataFrame([{'n': i, 'm': 10-i} for i in range(10)])
        df = ps.DataFrame(base_df)

        assertDataFrameEqualsPandas(df['n'] + 2 * df['m'],
                                    base_df['n'] + 2 * base_df['m'])
        assertDataFrameEqualsPandas((df['n'] - 1) // (2 ** (df['m'] % 3)),
                                    (base_df['n'] - 1) // (2 ** (base_df['m'] % 3)))  # noqa
        assertDataFrameEqualsPandas(abs(df['n']) // 5 & df['m'],
                                    abs(base_df['n']) // 5 & base_df['m'])
        assertDataFrameEqualsPandas(df['n'] | 0 ^ ~df['m'],
                                    base_df['n'] | 0 ^ ~base_df['m'])

    def test_aggregators(self):
        base_df = pd.DataFrame([{'m': i, 'n': 10-i} for i in range(0, 10)])
        df = ps.DataFrame(base_df)

        # Single-column aggregation
        self.assertEqual(df['n'].sum().compute(), base_df['n'].sum())
        self.assertEqual(df['n'].mean().compute(), base_df['n'].mean())
        self.assertEqual(df['n'].count().compute(), base_df['n'].count())
        self.assertEqual(df['n'].min().compute(), base_df['n'].min())
        self.assertEqual(df['n'].max().compute(), base_df['n'].max())
        self.assertEqual(df['n'].prod().compute(), base_df['n'].prod())
        self.assertEqual(df['n'].any().compute(), base_df['n'].any())
        self.assertEqual(df['n'].all().compute(), base_df['n'].all())

        # Multi-column aggregation
        assertDataFrameEqualsPandas(df.sum(), base_df.sum())
        assertDataFrameEqualsPandas(df.mean(), base_df.mean())
        assertDataFrameEqualsPandas(df.count(), base_df.count())
        assertDataFrameEqualsPandas(df.min(), base_df.min())
        assertDataFrameEqualsPandas(df.max(), base_df.max())
        assertDataFrameEqualsPandas(df.prod(), base_df.prod())
        assertDataFrameEqualsPandas(df.any(), base_df.any())
        assertDataFrameEqualsPandas(df.all(), base_df.all())

    def test_groupby(self):
        base_df = pd.DataFrame([
            {'a': str(i), 'b': str(j), 'c': 100*i, 'd': -j}
            for i in range(3) for j in range(3)
        ])
        df = ps.DataFrame(base_df)

        # Regular groupby
        res = df.groupby(['a', 'b'], as_index=False).sum()
        base_res = base_df.groupby(['a', 'b'], as_index=False).sum()
        assertDataFrameEqualsPandas(res, base_res)

        # groupby with group names in index
        res = df.groupby('a', as_index=True).prod()
        base_res = base_df.groupby('a', as_index=True).prod()
        assertDataFrameEqualsPandas(res, base_res)

        # Projection before aggregation
        res = df.groupby(['a', 'b'], as_index=False)['c'].count()
        base_res = base_df.groupby(['a', 'b'], as_index=False)['c'].count()
        assertDataFrameEqualsPandas(res, base_res)

        # Projection before aggregation with group names in index
        res = df.groupby(['a', 'b'], as_index=True)['c'].all()
        base_res = base_df.groupby(['a', 'b'], as_index=True)['c'].all()
        assertDataFrameEqualsPandas(res, base_res)

    def test_write_column(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        # Duplicate a column
        df['b'] = df['n']
        base_df['b'] = base_df['n']
        df['c'] = df['b'] * 2
        base_df['c'] = base_df['b'] * 2

        pd.testing.assert_index_equal(df.columns, base_df.columns)
        assertDataFrameEqualsPandas(df, base_df)

        # Write a constant column
        df['d'] = 10
        df['e'] = 'dummy'
        base_df['d'] = 10
        base_df['e'] = 'dummy'

        pd.testing.assert_index_equal(df.columns, base_df.columns)
        assertDataFrameEqualsPandas(df, base_df)

    def test_write_on_downstream_dataframe(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        # New columns
        selection = df[df['a'] != '4']
        selection['b'] = 10

        # Make new copy to avoid Pandas warning about writing to a slice
        expected = pd.DataFrame(base_df[base_df['a'] != '4'])
        expected['b'] = 10

        pd.testing.assert_index_equal(selection.columns, expected.columns)
        assertDataFrameEqualsPandas(selection, expected)

    def test_old_dependents_after_write(self):
        base_df = pd.DataFrame([{'n': i, 'a': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df, deep_copy=True)

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

        # This should trigger computation
        self.assertEqual(str(limit), str(base_limit))

        # All dependencies should also have cached results
        pd.testing.assert_frame_equal(merged.result, base_merged)
        pd.testing.assert_frame_equal(agg.result, base_agg)
        pd.testing.assert_frame_equal(ordered.result, base_ordered)
        pd.testing.assert_frame_equal(limit.result, base_limit)

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

        # This should trigger computation
        self.assertEqual(str(ordered), str(base_ordered))

        # All dependencies should also have cached results
        pd.testing.assert_frame_equal(merged.result, base_merged)
        pd.testing.assert_frame_equal(agg.result, base_agg)
        pd.testing.assert_frame_equal(ordered.result, base_ordered)


if __name__ == "__main__":
    unittest.main()
