import pandasql as ps
from pandasql.graph_utils import _get_dependency_graph, _filter_ancestors, \
    _get_ancestors_by_depth


class CostModel(object):
    def __init__(self, max_pandas_len=10,
                 max_pandas_input=500_000_000):  # 500 MB
        self.transfer_rate = 0  # GB/s
        self.required_pandas_ops = []
        self.required_sql_ops = []
        self.system_memory_size = 0  # TODO: automate finding this
        self.max_pandas_len = max_pandas_len
        self.max_pandas_input = max_pandas_input
        self.compute_path = []

    def should_offload(self, df):
        graph = _get_dependency_graph(df)

        for rule in OFFLOADING_RULES:
            result = rule(df, graph)
            if result is not None:  # Rule applied, and makes decision
                return result

        return False


##############################################################################
#                           Offloading Rules
##############################################################################


FARTHEST_DEPENDENCE = 1   # TODO: How much should we allow? Maybe 2-3?


def _join_then_restrict(df, graph):
    for node in graph.keys():
        # If result is already cached, no point in looking at it
        if node.result is not None:
            continue

        # TODO: Look at size of filter/limit for this decision?
        if isinstance(node, ps.core.Selection) or \
                isinstance(node, ps.core.Limit):
            join_ancestors = _filter_ancestors(node, graph, lambda x:
                                               isinstance(x, ps.core.Join),
                                               max_depth=FARTHEST_DEPENDENCE)
            if len(join_ancestors) > 0:
                return True


def _limit_output(df, graph):
    if isinstance(df, ps.core.Limit):
        return True


def _deep_dependency_graph(df, graph):
    ancestors_by_depth = _get_ancestors_by_depth(df, graph)
    depth = len(ancestors_by_depth)
    # TODO: move magic numbers like this one into some sort of config object
    if depth > 5:
        return True


def _fallback_operation(df, graph):
    # TODO: If some operations cannot be done without fallback, this method
    # chooses to run the entire query in Pandas. We should instead explore
    # running a subset of the query in SQLite, and then running Pandas-only
    # parts in Pandas.

    # Rule does not apply if the result of fallback is already cached
    fallbacks = _filter_ancestors(df, graph, lambda x: x.result is None and
                                  isinstance(x, ps.core.FallbackOperation),
                                  max_depth=None)
    if len(fallbacks) > 0:
        return False


OFFLOADING_RULES = [
    _fallback_operation,
    _join_then_restrict,
    _limit_output,
    _deep_dependency_graph,
    # TODO: more rules
]
