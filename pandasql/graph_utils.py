from queue import Queue
from collections import defaultdict


def _get_dependency_graph(df, on='duckdb'):
    dependencies = {}

    def add_dependencies(child):
        dependencies[child] = list(child.sources)
        if on == 'pandas':
            dependencies[child] += list(child.pandas_sources)

        for parent in dependencies[child]:
            if parent not in dependencies:
                add_dependencies(parent)

    add_dependencies(df)
    return dependencies


def _topological_sort(graph):
    # Reversed graph: node to children that depend on it
    parent_to_child = defaultdict(list)
    for child, parents in graph.items():
        for parent in parents:
            parent_to_child[parent].append(child)

    ready = Queue()  # Nodes that have 0 dependencies
    num_deps = {node: len(deps) for (node, deps) in graph.items()}
    for node, count in num_deps.items():
        if count == 0:
            ready.put(node)
    if ready.empty():
        raise ValueError('Cyclic dependency graph! Invalid computation.')

    results = []    # All nodes in order, with dependencies before dependents
    while not ready.empty():
        node = ready.get()
        results.append(node)

        # Update dependency counts for all dependents of node
        for child in parent_to_child[node]:
            num_deps[child] -= 1
            if num_deps[child] == 0:  # Child has no more dependencies
                ready.put(child)

    if len(results) != len(graph):
        raise RuntimeError('Expected {} nodes but could only sort {}'
                           .format(len(graph), len(results)))

    return results


def _filter_ancestors(df, graph, predicate, max_depth=None):
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
