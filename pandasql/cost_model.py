from pandasql.utils import _get_dependency_graph
import pandasql as ps

class CostModel(object):
    def __init__(self, max_pandas_len=10, max_pandas_input=500_000_000 ): #500 MB
        self.transfer_cost = 0 # GB/s
        self.required_pandas_ops = []
        self.required_sql_ops = []
        self.system_memory_size = 0
        self.max_pandas_len = max_pandas_len
        self.max_pandas_input = max_pandas_input
        self.compute_path = []

    def should_offload(self, df):

        # possible_executions = []
        graph = _get_dependency_graph(df)
        query_height = len(graph)
        estimated_input = sum([g._input_size for g in graph])

        limits = [g.n for g in graph if isinstance(g, ps.Core.Limit)]
        filters = [g.criterion for g in graph if isinstance(g, ps.Core.Selection)]
        aggregations = [g.agg for g in graph if isinstance(g, ps.Core.Aggregator)]
        joins = [g.left_keys for g in graph if isinstance(g, ps.Core.Join)]


        # print(num_limits, num_filters, num_aggregations, num_joins)
        print([type(g) for g in graph])

        print(estimated_input, self.max_pandas_input)

        # if query_height > self.max_pandas_len:
        #     print('offloading from query height')
        #     return True

        if estimated_input > self.max_pandas_input:
            print('offloading from input size')
            return True

        if len(aggregations) + len(joins) > 2:
            print('offloading from agg/join')
            return True

        if len(limits) > 0 and min(limits) < 10000:
            print('offloading from limit')
            return True


        else:
            return False
        # execution_estimates = []
        # for pe in possible_executions:
            # cost = 0 # some formula
            # execution time + possible data transfer time
            # we in
            # execution_estimates.append(pe, cost)

        # best_plan = min(execution_estimates, key = lambda t: t[1])

