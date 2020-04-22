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

    def test_arithmetic(self):
        return NotImplemented
