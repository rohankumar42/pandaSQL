import unittest
import pandas as pd
import pandasql as ps
from .utils import compute_all_ancestors


class TestMemoryPredictor(unittest.TestCase):
    # TODO(important): Add tests for all objects' memory predictors!

    def checkMemoryPrediction(self, df: ps.DataFrame, base_df: pd.DataFrame,
                              delta=0.1):
        pd_mem = base_df.memory_usage(deep=True, index=True)
        if isinstance(pd_mem, pd.Series):
            pd_mem = pd_mem.sum()

        compute_all_ancestors(df)
        ps_mem = df._predict_memory_from_sources()

        self.assertAlmostEqual(pd_mem, ps_mem, delta=delta * pd_mem)

    def test_simple_memory_prediction(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)
        self.checkMemoryPrediction(df, base_df, delta=0.0)

    def test_projection_memory_prediction(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_proj = base_df.n
        proj = df.n
        self.checkMemoryPrediction(proj, base_proj, delta=0.0)

    def test_criterion_memory_prediction(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_criterion = base_df.n > 4
        criterion = df.n > 4
        self.checkMemoryPrediction(criterion, base_criterion, delta=0.0)

    def test_selection_memory_prediction(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_sel = base_df[base_df.n > 4]
        sel = df[df.n > 4]
        self.checkMemoryPrediction(sel, base_sel)

    def test_limit_memory_prediction(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        df = ps.DataFrame(base_df)

        base_limit = base_df[:25]
        limit = df[:25]
        self.checkMemoryPrediction(limit, base_limit)

    def test_order_by_memory_prediction(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        base_ordered = base_df.sort_values(by='n', ascending=False)
        ordered = df.sort_values(by='n', ascending=False)
        self.checkMemoryPrediction(ordered, base_ordered)

    def test_join_memory_prediction(self):
        return NotImplemented

    def test_union_memory_prediction(self):
        return NotImplemented

    def test_aggregate_memory_prediction(self):
        return NotImplemented

    def test_grouped_aggregate_memory_prediction(self):
        return NotImplemented

    def test_arithmetic_memory_prediction(self):
        return NotImplemented

    def test_write_memory_prediction(self):
        return NotImplemented


if __name__ == "__main__":
    unittest.main()
