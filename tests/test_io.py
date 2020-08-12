import unittest
import os

import pandasql as ps
import pandas as pd
import numpy as np

from .utils import assertDataFrameEqualsPandas


class TestIO(unittest.TestCase):

    def setUp(self):
        self.FILE_NAME = '.test_csv'
        ps.offloading_strategy('ALWAYS')
        arr = np.random.randint(low=0, high=100_000_000, size=(1000, 16))
        np.savetxt(self.FILE_NAME, arr, delimiter=',', fmt='%i',
                   header='c0,c1,c2,c3,c4,c5,c6,c7,c8,c9,c10,c11,c12,c13,c14,c15',  # noqa
                   comments='')
        self.addCleanup(os.remove, self.FILE_NAME)

    def test_loading_csv(self):
        df = ps.read_csv(self.FILE_NAME)
        base_df = pd.read_csv(self.FILE_NAME)

        df['c0'] += 1
        base_df['c0'] += 1

        df.compute()
        assertDataFrameEqualsPandas(df, base_df)

    def test_loading_csv_sql(self):
        df = ps.read_csv(self.FILE_NAME, sql_load=True)
        base_df = pd.read_csv(self.FILE_NAME)

        df['c0'] += 1
        base_df['c0'] += 1

        df.compute()
        assertDataFrameEqualsPandas(df, base_df)

    def test_loading_csv_sql_chunk(self):
        ps.io.MEMORY_THRESHOLD = 1

        df = ps.read_csv(self.FILE_NAME)
        base_df = pd.read_csv(self.FILE_NAME)

        df['c0'] += 1
        base_df['c0'] += 1

        df.compute()
        assertDataFrameEqualsPandas(df, base_df)


if __name__ == "__main__":
    unittest.main()
