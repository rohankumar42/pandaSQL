import unittest
import pandas as pd
import pandasql as ps
from pandasql.utils import _get_dependency_graph, _topological_sort


class TestUtils(unittest.TestCase):

    def test_get_dependency_graph(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = ps.DataFrame(base_df_2)
        joined = df_1.join(df_2, on='n')

        graph = _get_dependency_graph(joined)
        self.assertIn(df_1, graph)
        self.assertIn(df_2, graph)
        self.assertIn(joined, graph)
        self.assertIn(df_1, set(graph[joined]))
        self.assertIn(df_2, set(graph[joined]))
        self.assertEqual(len(graph[joined]), 2)
        self.assertEqual(len(graph[df_1]), 0)
        self.assertEqual(len(graph[df_2]), 0)

    def test_topological_sort(self):
        base_df_1 = pd.DataFrame([{'n': i, 's1': str(i*2)} for i in range(10)])
        df_1 = ps.DataFrame(base_df_1)
        base_df_2 = pd.DataFrame([{'n': i, 's2': str(i*2)} for i in range(10)])
        df_2 = ps.DataFrame(base_df_2)
        joined = df_1.join(df_2, on='n')

        graph = _get_dependency_graph(joined)
        ordered = _topological_sort(graph)
        self.assertEqual(ordered[0].name, df_1.name)
        self.assertEqual(ordered[1].name, df_2.name)
        self.assertEqual(ordered[2].name, joined.name)


if __name__ == "__main__":
    unittest.main()
