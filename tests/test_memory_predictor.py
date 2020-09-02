import unittest

import pandas as pd

import pandasql as ps


class TestCostModel(unittest.TestCase):

    def setUp(self):
        self.THRESHOLD = 0.2

    def test_simple_dataframe_usage(self):
        base_df = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        pd_mem = base_df.memory_usage(deep=True, index=True).sum()
        ps_mem = df._predict_memory_from_sources()
        self.assertAlmostEqual(pd_mem, ps_mem, delta=self.THRESHOLD * pd_mem)

    def test_projection_dataframe_usage(self):
        base_df = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_df_1 = base_df.n
        df_1 = df.n

        pd_mem = base_df_1.memory_usage(deep=True, index=True)
        ps_mem = df_1._predict_memory_from_sources()
        self.assertAlmostEqual(pd_mem, ps_mem, delta=self.THRESHOLD * pd_mem)

    def test_criterion_dataframe_usage(self):
        base_df = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_sel = base_df.n > 4
        sel = df.n > 4

        pd_mem = base_sel.memory_usage(deep=True, index=True)
        ps_mem = sel._predict_memory_from_sources()
        self.assertAlmostEqual(pd_mem, ps_mem, delta=self.THRESHOLD * pd_mem)

if __name__ == "__main__":
    unittest.main()
