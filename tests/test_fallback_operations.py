import unittest

import pandas as pd
import pandasql as ps
from .utils import assertDataFrameEqualsPandas


class TestFallbackOperations(unittest.TestCase):

    def setUp(self):
        ps.offloading_strategy('NEVER')

    def test_nlargest_nsmallest(self):
        base_df = pd.DataFrame([{'n': i, 's': str(i*2)} for i in range(30)])
        df = ps.DataFrame(base_df)

        base_largest = base_df.nlargest(n=10, columns='n')
        largest = df.nlargest(n=10, columns='n')
        self.assertIsInstance(largest, ps.core.FallbackOperation)
        assertDataFrameEqualsPandas(largest, base_largest)

        base_smallest = base_df.nsmallest(n=5, columns='n')
        smallest = df.nsmallest(n=5, columns='n')
        self.assertIsInstance(smallest, ps.core.FallbackOperation)
        assertDataFrameEqualsPandas(smallest, base_smallest)


if __name__ == "__main__":
    unittest.main()
