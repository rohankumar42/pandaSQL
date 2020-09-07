import unittest
import pandas as pd
import pandasql as ps
from .utils import compute_all_ancestors


class TestMemoryPredictor(unittest.TestCase):
    # TODO(important): Add tests for all objects' memory predictors!

    def setUp(self):
        ps.offloading_strategy('NEVER')

    def checkMemoryPrediction(self, df: ps.DataFrame, delta=0.1):
        compute_all_ancestors(df)
        predicted = df._predict_memory_from_sources()

        df._compute_pandas()
        actual = df._cached_result.memory_usage(deep=True, index=True)
        if isinstance(actual, pd.Series):
            actual = actual.sum()

        self.assertAlmostEqual(actual, predicted, delta=delta * actual)

    def test_projection_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        proj = df.n
        self.checkMemoryPrediction(proj, delta=0.0)

    def test_criterion_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        criterion = df.n > 4
        self.checkMemoryPrediction(criterion, delta=0.0)

    def test_selection_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        sel = df[df.n > 4]
        self.checkMemoryPrediction(sel)

    def test_limit_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        limit = df[:25]
        self.checkMemoryPrediction(limit)

    def test_order_by_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        ordered = df.sort_values(by='n', ascending=False)
        self.checkMemoryPrediction(ordered)

    def test_merge_memory_prediction(self):
        df_1 = ps.DataFrame([{'n': i, 's1': str(i*2)}
                             for i in range(10, 10000, 5)])
        df_2 = ps.DataFrame([{'n': i, 's2': str(i*2)}
                             for i in range(20000, 4000, -15)])
        merged = df_1.merge(df_2, on='n')
        self.checkMemoryPrediction(merged)

    def test_union_memory_prediction(self):
        df_1 = ps.DataFrame([{'n': i, 's': str(i)}
                             for i in range(10, 1000, 10)])
        df_2 = ps.DataFrame([{'n': i, 's': str(i)}
                             for i in range(200, 400, 1)])
        union = ps.concat([df_1, df_2])
        self.checkMemoryPrediction(union)

    def test_aggregate_memory_prediction(self):
        return NotImplemented

    def test_grouped_aggregate_memory_prediction(self):
        return NotImplemented

    def test_arithmetic_constant_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': i*2} for i in range(100)])
        added = df.n + 1
        self.checkMemoryPrediction(added)

    def test_arithmetic_projection_memory_prediction(self):
        df = ps.DataFrame([{'n': i, 's': i*2} for i in range(100)])
        df.n += df.s
        self.checkMemoryPrediction(df)

    def test_write_memory_prediction(self):
        return NotImplemented


if __name__ == "__main__":
    unittest.main()
