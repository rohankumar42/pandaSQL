from pandasql.utils import _get_dependency_graph
import pandasql as ps


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


MAX_DEPENDENCY_DISTANCE = 1   # TODO: How much should we allow? Maybe 2-3?


def _out_of_memory(df, graph):
    # TODO: How should we estimate memory usage of a query?
    return NotImplemented


def _join_then_restrict(df, graph):
    for node in graph.keys():
        # If result is already cached, no point in looking at it
        if node.result is not None:
            continue

        # TODO: Look at size of filter/limit for this decision?
        if isinstance(node, ps.core.Selection) or \
                isinstance(node, ps.core.Limit):
            join_ancestors = _filter_ancestors(node, graph, lambda x:
                                               isinstance(x, ps.core.Join))
            if len(join_ancestors) > 0:
                return True


def _limit_output(df, graph):
    if isinstance(df, ps.core.Limit):
        return True


def _deep_dependency_graph(df, graph):
    ancestors_by_depth = _get_ancestors_by_depth(df, graph)
    depth = len(ancestors_by_depth)
    if depth > 5:
        return True


def _fallback_operation(df, graph):
    # Rule does not apply if the result of fallback is already cached
    fallbacks = _filter_ancestors(df, graph, lambda x: x.result is None and
                                  isinstance(x, ps.core.FallbackOperation),
                                  max_depth=None)
    if len(fallbacks) > 0:
        return False


OFFLOADING_RULES = [
    _fallback_operation,
    # _out_of_memory,
    _join_then_restrict,
    _limit_output,
    _deep_dependency_graph,
    # TODO: more rules
]


##############################################################################
#                       Common Utility Functions
##############################################################################


def _filter_ancestors(df, graph, predicate,
                      max_depth=MAX_DEPENDENCY_DISTANCE):
    ancestors_by_depth = _get_ancestors_by_depth(df, graph, max_depth)
    filtered = []
    for ancestors in ancestors_by_depth.values():
        filtered.extend(filter(predicate, ancestors))
    return filtered


def _get_ancestors_by_depth(df, graph, max_depth=None):
    max_depth = len(graph) if max_depth is None else max_depth

    visited = {df}
    ancestors_by_depth = {0: {df}}

    depth = 0
    while depth < max_depth and \
            len(ancestors_by_depth[depth]) > 0:  # There are unexplored nodes

        cur_layer = ancestors_by_depth[depth]
        depth += 1
        ancestors_by_depth[depth] = set()
        for cur in cur_layer:
            for neighbor in graph[cur]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    ancestors_by_depth[depth].add(neighbor)

    return ancestors_by_depth