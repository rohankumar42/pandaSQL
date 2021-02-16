import unittest
import os
import psutil
from tempfile import mkstemp

import pandasql as ps
import pandas as pd
import numpy as np

from .utils import assertDataFrameEqualsPandas


class TestIO(unittest.TestCase):

    def setUp(self):
        self.FILE_NAME = mkstemp('.test_csv')[1]
        ps.offloading_strategy('ALWAYS')
        arr = np.random.randint(low=0, high=100_000_000, size=(25_000, 2))
        np.savetxt(self.FILE_NAME, arr, delimiter=',', fmt='%i',
                   header='c0,c1', comments='')
        self.addCleanup(os.remove, self.FILE_NAME)

    def test_loading_csv(self):
        df = ps.read_csv(self.FILE_NAME)
        base_df = pd.read_csv(self.FILE_NAME)

        df['c0'] += 1
        base_df['c0'] += 1

        assertDataFrameEqualsPandas(df, base_df)

    def test_loading_csv_sqlite(self):
        df = ps.read_csv(self.FILE_NAME, sql_load=True)
        base_df = pd.read_csv(self.FILE_NAME)

        df['c0'] += 1
        base_df['c0'] += 1

        assertDataFrameEqualsPandas(df, base_df)

    def test_loading_csv_in_chunks(self):
        memory_thresh = 10 ** 4
        new_factor = memory_thresh / psutil.virtual_memory().available
        old_factor = ps.memory_utils.SAFETY_FACTOR
        ps.memory_utils.SAFETY_FACTOR = new_factor

        df = ps.read_csv(self.FILE_NAME)
        base_df = pd.read_csv(self.FILE_NAME)

        ps.memory_utils.SAFETY_FACTOR = old_factor

        df['c0'] += 1
        base_df['c0'] += 1

        assertDataFrameEqualsPandas(df, base_df)


if __name__ == "__main__":
    unittest.main()
