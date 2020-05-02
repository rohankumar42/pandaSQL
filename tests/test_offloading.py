import unittest

import pandas as pd
import pandasql as ps
from utils import assertDataFrameEqualsPandas


class TestPandasExecution(unittest.TestCase):

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


if __name__ == "__main__":
    unittest.main()
