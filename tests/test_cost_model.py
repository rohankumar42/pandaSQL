import unittest

import pandasql as ps
from pandasql.core import COST_MODEL


class TestCostModel(unittest.TestCase):

    def test_offloading_rule_out_of_memory(self):
        return NotImplemented

    def test_offloading_rule_join_then_restrict(self):
        df_1 = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(100)])
        df_2 = ps.DataFrame([{'n': i, 't': str(i*4)} for i in range(100)])

        join = df_1.merge(df_2, on='n')
        filtered = join[join['s'] + join['t'] < 50]
        limit = join[:20]

        self.assertFalse(COST_MODEL.should_offload(df_1))
        self.assertFalse(COST_MODEL.should_offload(df_2))
        self.assertTrue(COST_MODEL.should_offload(filtered))
        self.assertTrue(COST_MODEL.should_offload(limit))

        limit.compute()
        selection = limit['s']
        # No pending join-limit operations for selection
        self.assertFalse(COST_MODEL.should_offload(selection))

    def test_offloading_rule_limit_output(self):
        df = ps.DataFrame([{'n': i, 's': str(i % 2)} for i in range(100)])

        filtered = df[df['n'] > 25]
        limit = filtered.head(5)

        self.assertFalse(COST_MODEL.should_offload(filtered))
        self.assertTrue(COST_MODEL.should_offload(limit))

    def test_offloading_rule_deep_dependency_graph(self):
        depth = 10
        size = 10 ** 5
        step = size // depth

        df = ps.DataFrame([{'n': i, 's': str(i*2)} for i in range(size)])
        descendants = [df]

        for d in range(step, size, step):
            parent = descendants[-1]
            child = parent[parent['n'] > d]
            descendants.append(child)

        self.assertFalse(COST_MODEL.should_offload(df))
        self.assertTrue(COST_MODEL.should_offload(descendants[-1]))

    def test_offloading_fallback_operation(self):
        ps.offloading_strategy('BEST')

        df = ps.DataFrame([{'n': i, 's': str(i % 2)} for i in range(100)])

        largest = df.nlargest(10, 'n')
        limit = largest[:3]

        self.assertFalse(COST_MODEL.should_offload(largest))
        self.assertFalse(COST_MODEL.should_offload(limit))

        largest.compute()
        # No more pending fallback operations now
        self.assertTrue(COST_MODEL.should_offload(limit))


if __name__ == "__main__":
    unittest.main()
