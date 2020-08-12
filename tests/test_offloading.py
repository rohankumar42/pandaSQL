import unittest

import pandas as pd
import pandasql as ps
from .utils import assertDataFrameEqualsPandas


class TestOffloading(unittest.TestCase):

    def test_result_synchronization_simple(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        df = ps.DataFrame(base_df)

        selection = df[df['n'] >= 5]
        base_selection = base_df[base_df['n'] >= 5]
        limit = selection[:3]
        base_limit = base_selection[:3]

        # Compute selection on Pandas
        ps.offloading_strategy('NEVER')
        self.assertIsNone(selection.result)
        selection.compute()
        assertDataFrameEqualsPandas(selection, base_selection)

        # Run dependent operation on SQLite
        ps.offloading_strategy('ALWAYS')
        self.assertIsNone(limit.result)
        self.assertEqual(limit.sql(),
                         f'WITH {selection.name} AS ({selection.sql()}) '
                         f'SELECT * FROM {selection.name} LIMIT 3')
        limit.compute()
        assertDataFrameEqualsPandas(limit, base_limit)

    def test_run_with_missing_dependencies_sqlite(self):
        ps.offloading_strategy('ALWAYS')

        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        base_selection = base_df[base_df['n'] >= 5]

        # Should run on SQLite since original data was offloaded
        df = ps.DataFrame(base_df, offload=True)
        selection = df[df['n'] >= 5]
        assertDataFrameEqualsPandas(selection, base_selection)

        # Should not run on SQLite since original data was not offloaded
        df = ps.DataFrame(base_df, offload=False)
        selection = df[df['n'] >= 5]
        self.assertRaises(RuntimeError, lambda: selection.compute())

    def test_run_with_missing_dependencies_pandas(self):
        ps.offloading_strategy('NEVER')

        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(10)])
        base_selection = base_df[base_df['n'] >= 5]

        # Should run on Pandas since original data exists
        df = ps.DataFrame(base_df)
        selection = df[df['n'] >= 5]
        assertDataFrameEqualsPandas(selection, base_selection)

        # Should not run on Pandas since original data does not exist
        df._cached_result = None
        selection = df[df['n'] >= 5]
        self.assertRaises(RuntimeError, lambda: selection.compute())


if __name__ == "__main__":
    unittest.main()
