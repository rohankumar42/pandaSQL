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
        return NotImplemented

    def test_aggregators(self):
        return NotImplemented

    def test_groupby(self):
        return NotImplemented

    def test_write_column(self):
        return NotImplemented

    def test_complex_read_query(self):
        return NotImplemented

    def test_complex_write_query(self):
        return NotImplemented
