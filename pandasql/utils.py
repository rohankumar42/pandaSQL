# import uuid
from queue import Queue
from collections import defaultdict


# TODO: add more supported types
SUPPORTED_TYPES = [int, float, str, list]

# TODO: switch to uuids when done testing
COUNT = 0


def _is_supported_constant(x):
    return any(isinstance(x, t) for t in SUPPORTED_TYPES)


def _new_name():
    # name = uuid.uuid4().hex
    global COUNT
    name = 'T' + str(COUNT)
    COUNT += 1
    return name


def _get_dependency_graph(table):
    dependencies = {}

    def add_dependencies(child):
        dependencies[child] = list(child.sources)
        for parent in dependencies[child]:
            if parent not in dependencies:
                add_dependencies(parent)

    add_dependencies(table)
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
