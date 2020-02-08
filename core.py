# import uuid
import sqlite3
import pandas as pd
from queue import Queue
from collections import defaultdict

# TODO: add more supported types
SUPPORTED_TYPES = [int, float, str]
# TODO: switch to uuids when done testing
COUNT = 0
SQL_CON = sqlite3.connect(":memory:")


def _is_supported_constant(x):
    return any(isinstance(x, t) for t in SUPPORTED_TYPES)


def _get_dependency_graph(table):
    assert isinstance(table, BaseTable)
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


def _define_dependencies(table):
    graph = _get_dependency_graph(table)
    ordered_deps = _topological_sort(graph)

    common_table_exprs = [
        '{} AS ({})'.format(t.name, t.sql(dependencies=False))
        for t in ordered_deps if not t.is_base_table and t is not table
    ]

    if len(common_table_exprs) > 0:
        return 'WITH ' + ', '.join(common_table_exprs)
    else:
        return None


class BaseThunk(object):
    # TODO: maybe think of a better name than thunk
    def __init__(self, name=None):
        # self.name = name or uuid.uuid4().hex
        global COUNT
        self.name = name or 'T' + str(COUNT)
        COUNT += 1
        self.sources = []

    def sql(self, dependencies=True):
        raise TypeError('Objects of type {} cannot be converted to a SQL query'
                        .format(type(self)))

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.name)


class BaseTable(BaseThunk):
    # TODO: cache computed results
    def __init__(self, name=None):
        super().__init__(name=name)
        self.base_table = self

    @property
    def is_base_table(self):
        return self.base_table is self

    @classmethod
    def from_pandas(cls, df: pd.DataFrame, name=None):
        table = cls(name=name)
        df.to_sql(name=table.name, con=SQL_CON)
        return table

    def compute(self):
        query = self.sql()
        return pd.read_sql_query(query, con=SQL_CON)

    def __getitem__(self, x):
        if isinstance(x, str) or isinstance(x, list):  # TODO: check valid cols
            return Projection(self, x)
        elif isinstance(x, BaseCriterion):
            return Selection(self, x)
        elif isinstance(x, int):
            raise NotImplementedError('TODO: iloc/loc based access')
        else:
            raise TypeError('Unsupported indexing type {}'.format(type(x)))

    def join(self, other, on=None, **args):
        """TODO: support other pandas join arguments"""
        assert(isinstance(other, BaseTable))
        if on is None:
            raise NotImplementedError('TODO: implement cross join')
        else:
            assert(isinstance(on, str))
            return Join(self, other, self[on] == other[on])

    def __eq__(self, other):
        return self._comparison(other, Equal)

    def __ne__(self, other):
        return self._comparison(other, NotEqual)

    def __lt__(self, other):
        return self._comparison(other, LessThan)

    def __le__(self, other):
        return self._comparison(other, LessThanOrEqual)

    def __gt__(self, other):
        return self._comparison(other, GreaterThan)

    def __ge__(self, other):
        return self._comparison(other, GreaterThanOrEqual)

    def _comparison(self, other, how_class):
        return how_class(self._make_projection_or_constant(self),
                         self._make_projection_or_constant(other))

    def __hash__(self):
        return super().__hash__()

    @staticmethod
    def _make_projection_or_constant(x):
        if isinstance(x, Projection):
            return x
        elif _is_supported_constant(x):
            return Constant(x)
        else:
            raise TypeError('Only constants and Projections are accepted')


class Projection(BaseTable):
    def __init__(self, source, col, name=None):
        assert(isinstance(source, BaseTable))

        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, list):
            cols = col
        else:
            raise TypeError('col must be of type str or list, but found {}'
                            .format(type(col)))

        super().__init__(name=name)
        self.sources = [source]
        self.base_table = source.base_table
        self.cols = cols

    def __str__(self):
        attrs = ', '.join('{}.{}'.format(self.sources[0].name, col)
                          for col in self.cols)
        if len(self.cols) > 1:
            attrs = '({})'.format(attrs)
        return attrs

    def sql(self, dependencies=True):
        query = []

        if dependencies:
            common_table_expr = _define_dependencies(self)
            if common_table_expr is not None:
                query.append(common_table_expr)

        query.append('SELECT {} FROM {}'.format(', '.join(self.cols),
                                                self.sources[0].name))

        return ' '.join(query)


class Selection(BaseTable):
    def __init__(self, source, criterion, name=None):
        assert(isinstance(source, BaseTable))
        assert(isinstance(criterion, BaseCriterion))

        # TODO: have well thought out type checking

        super().__init__(name=name)
        self.sources = [source]
        self.base_table = source.base_table
        self.criterion = criterion

    def __str__(self):
        return 'Select({}, {})'.format(self.sources[0], self.criterion)

    def sql(self, dependencies=True):
        query = []

        if dependencies:
            common_table_expr = _define_dependencies(self)
            if common_table_expr is not None:
                query.append(common_table_expr)

        query.append('SELECT * FROM {} WHERE {}'.format(self.sources[0].name,
                                                        self.criterion))

        return ' '.join(query)


class Join(BaseTable):
    def __init__(self, source_1, source_2, criterion, name=None):
        assert(isinstance(source_1, BaseTable))
        assert(isinstance(source_2, BaseTable))
        assert(isinstance(criterion, BaseCriterion))

        # TODO: have well thought out type checking

        super().__init__(name=name)
        self.sources = [source_1, source_2]
        self.base_tables = [source_1.base_table, source_2.base_table]
        self.criterion = criterion

    def __str__(self):
        source_1, source_2 = self.sources
        return 'Join({}, {}, {})'.format(source_1, source_2,
                                         self.criterion)

    def sql(self, dependencies=True):
        query = []

        if dependencies:
            common_table_expr = _define_dependencies(self)
            if common_table_expr is not None:
                query.append(common_table_expr)

        query.append('SELECT * FROM {} JOIN {} ON {}'
                     .format(self.sources[0].name, self.sources[1].name,
                             self.criterion))

        return ' '.join(query)


class Constant(BaseThunk):

    def __init__(self, value, name=None):
        super().__init__(name=name)
        if not _is_supported_constant(value):
            raise TypeError('Unsupported type {}'.format(type(value)))
        self.value = value

    def __str__(self):
        return str(self.value)


class BaseCriterion(BaseThunk):
    def __init__(self, operation, source_1, source_2, name=None):
        assert(isinstance(source_1, Projection) or
               isinstance(source_1, Constant))
        assert(isinstance(source_2, Projection) or
               isinstance(source_2, Constant))

        super().__init__(name=name)
        self.operation = operation
        self.sources = [source_1, source_2]
        # if isinstance(source_1, Projection):
        # self.sources.append(source_1.base_table)
        # if isinstance(source_2, Projection):
        # self.sources.append(source_2.base_table)

    def __str__(self):
        source_1, source_2 = self.sources
        return '{} {} {}'.format(source_1, self.operation, source_2)


class Equal(BaseCriterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('=', source_1, source_2, name=name)


class NotEqual(BaseCriterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<>', source_1, source_2, name=name)


class LessThan(BaseCriterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<', source_1, source_2, name=name)


class LessThanOrEqual(BaseCriterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<=', source_1, source_2, name=name)


class GreaterThan(BaseCriterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('>', source_1, source_2, name=name)


class GreaterThanOrEqual(BaseCriterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('>=', source_1, source_2, name=name)


class DataFrame(object):

    def __init__(self):
        raise NotImplementedError("TODO: figure out if this is needed")
