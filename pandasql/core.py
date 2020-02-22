# import uuid
import sqlite3
import pandas as pd
from typing import List

from pandasql.utils import _is_supported_constant, _get_dependency_graph, \
    _topological_sort

# TODO: switch to uuids when done testing
COUNT = 0
SQL_CON = sqlite3.connect(":memory:")


class BaseThunk(object):
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


class Constant(BaseThunk):

    def __init__(self, value, name=None):
        super().__init__(name=name)
        if not _is_supported_constant(value):
            raise TypeError('Unsupported type {}'.format(type(value)))
        self.value = value

    def __str__(self):
        return str(self.value)


class Criterion(BaseThunk):
    def __init__(self, operation, source_1, source_2=None,
                 name=None, simple=True):
        # Simple criteria depend on comparisons between columns and/or
        # constants, whereas compound criteria depend on smaller criteria
        if simple:
            assert(isinstance(source_1, Projection) or
                   isinstance(source_1, Constant))
            if source_2 is not None:
                assert(isinstance(source_2, Projection) or
                       isinstance(source_2, Constant))
        else:
            assert(isinstance(source_1, Criterion))
            if source_2 is not None:
                assert(isinstance(source_2, Criterion))

        super().__init__(name=name)
        self.operation = operation
        self.sources = [source_1]
        if source_2 is not None:
            self.sources.append(source_2)

    def __str__(self):
        '''Default for binary Criterion objects'''
        return '{} {} {}'.format(self._source_to_str(self.sources[0]),
                                 self.operation,
                                 self._source_to_str(self.sources[1]))

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    @staticmethod
    def _source_to_str(source):
        if isinstance(source, Projection):
            attrs = ', '.join('{}.{}'.format(source.sources[0].name, col)
                              for col in source.cols)
            if len(source.cols) > 1:
                attrs = '({})'.format(attrs)
            return attrs

        elif isinstance(source, Constant) or isinstance(source, Criterion):
            return str(source)

        else:
            raise TypeError("Unexpected source type {}".format(type(source)))


class DataFrame(BaseThunk):
    def __init__(self, data=None, name=None, sources=None, base_tables=None):
        super().__init__(name=name)
        self.sources = sources or []
        self.base_tables = base_tables or [self]
        self.result = None
        df = None

        # If data provided, result is already ready
        if isinstance(data, dict) or isinstance(data, list):
            df = pd.DataFrame(data)
            self.result = df
        elif isinstance(data, pd.DataFrame):
            df = data
            self.result = df
        elif isinstance(data, DataFrame):
            df = data.compute()
            self.result = df
        elif data is None:
            pass
        else:
            raise TypeError('Cannot create table from object of type {}'
                            .format(type(data)))

        # Offload dataframe to SQLite
        if df is not None and len(df) > 0:
            df.to_sql(name=self.name, con=SQL_CON, index=False)

    @property
    def is_base_table(self):
        return len(self.base_tables) == 1 and self.base_tables[0] is self

    def require_result(func):  # noqa
        '''Decorator for functions that require results to be ready'''

        def result_ensured(self, *args, **kwargs):
            self.compute()
            return func(self, *args, **kwargs)

        return result_ensured

    def compute(self):

        if self.result is None:
            query = self.sql(dependencies=True)
            self.result = pd.read_sql_query(query, con=SQL_CON)
            # TODO: directly save results in SQLite instead
            self.result.to_sql(name=self.name, con=SQL_CON, index=False)

        return self.result

    def _create_sql_query(self):
        raise NotImplementedError("DataFrame objects don't need a SQL query")

    def sql(self, dependencies=True):
        query = []

        if dependencies:
            common_table_expr = _define_dependencies(self)
            if common_table_expr is not None:
                query.append(common_table_expr)

        query.append(self._create_sql_query())

        return ' '.join(query)

    def __getitem__(self, x):
        if isinstance(x, str) or isinstance(x, list):  # TODO: check valid cols
            return Projection(self, x)
        elif isinstance(x, Criterion):
            return Selection(self, x)
        elif isinstance(x, slice):
            if x.start is not None or x.step is not None:
                raise ValueError('Only slices of the form df[:n] are accepted')
            return self if x.stop is None else Limit(self, n=x.stop)
        else:
            raise TypeError('Unsupported indexing type {}'.format(type(x)))

    def head(self, n=5):
        return self[:n]

    @property
    def columns(self):
        # Quick hack: compute the first row of the result
        result = self[:1].compute()
        return result.columns

    @require_result
    def __str__(self):
        return str(self.result)

    @require_result
    def __len__(self):
        return len(self.result)

    def sort_values(self, by, ascending=True):
        return OrderBy(self, cols=by, ascending=ascending)

    def join(self, other, on=None):
        """TODO: support other pandas join arguments"""
        if on is None:
            raise NotImplementedError('Joins without key(s) are not supported')
        else:
            return Join(self, other, on)

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

    @require_result
    def to_csv(self, *args, **kwargs):
        return self.result.to_csv(*args, **kwargs)

    @require_result
    def to_json(self, *args, **kwargs):
        return self.result.to_json(*args, **kwargs)

    @require_result
    def to_numpy(self, *args, **kwargs):
        return self.result.to_numpy(*args, **kwargs)

    @require_result
    def to_pickle(self, *args, **kwargs):
        return self.result.to_pickle(*args, **kwargs)

    @require_result
    def _repr_html_(self, *args, **kwargs):
        '''For being pretty in Jupyter noteboks'''
        return self.result._repr_html_(*args, **kwargs)

    @require_result
    def _repr_latex_(self, *args, **kwargs):
        return self.result._repr_latex_(*args, **kwargs)

    @require_result
    def _repr_data_resource_(self, *args, **kwargs):
        return self.result._repr_data_resource_(*args, **kwargs)

    @require_result
    def _repr_fits_horizontal_(self, *args, **kwargs):
        return self.result._repr_fits_horizontal_(*args, **kwargs)

    @require_result
    def _repr_fits_vertical_(self, *args, **kwargs):
        return self.result._repr_fits_vertical_(*args, **kwargs)


class Projection(DataFrame):
    def __init__(self, source: DataFrame, col, name=None):

        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, list):
            cols = col
        else:
            raise TypeError('col must be of type str or list, but found {}'
                            .format(type(col)))

        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)
        self.cols = cols

    def _create_sql_query(self):
        return 'SELECT {} FROM {}'.format(', '.join(self.cols),
                                          self.sources[0].name)


class Selection(DataFrame):
    def __init__(self, source: DataFrame, criterion: Criterion, name=None):
        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)
        self.criterion = criterion

    def _create_sql_query(self):
        return 'SELECT * FROM {} WHERE {}'.format(self.sources[0].name,
                                                  self.criterion)


class OrderBy(DataFrame):
    def __init__(self, source: DataFrame, cols, ascending=True, name=None):

        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(ascending, bool):
            ascending = [ascending]
        if len(cols) != len(ascending):
            raise ValueError("cols and ascending must be equal lengths, "
                             "but found len(cols)={} and len(ascending)={}"
                             .format(len(cols), len(ascending)))

        self.cols = cols
        self.ascending = ascending

    def _create_sql_query(self):
        order_by = [
            '{}.{} {}'.format(self.sources[0].name, col,
                              'ASC' if asc else 'DESC')
            for col, asc in zip(self.cols, self.ascending)
        ]

        return 'SELECT * FROM {} ORDER BY {}' \
            .format(self.sources[0].name, ', '.join(order_by))


class Join(DataFrame):
    def __init__(self, source_1: DataFrame, source_2: DataFrame,
                 join_keys, name=None):
        super().__init__(name=name, sources=[source_1, source_2],
                         base_tables=source_1.base_tables +
                         source_2.base_tables)
        if isinstance(join_keys, str):
            join_keys = [join_keys]
        self.join_keys = join_keys

    def _create_sql_query(self):
        return 'SELECT * FROM {} JOIN {} USING ({})' \
            .format(self.sources[0].name, self.sources[1].name,
                    ','.join(self.join_keys))


class Union(DataFrame):
    def __init__(self, sources: List[DataFrame], name=None):
        base_tables = list({base for s in sources for base in s.base_tables})
        super().__init__(name=name, sources=sources, base_tables=base_tables)

    def _create_sql_query(self):
        return ' UNION ALL '.join('SELECT * FROM {}'.format(source.name)
                                  for source in self.sources)


class Limit(DataFrame):
    def __init__(self, source: DataFrame, n, name=None):

        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)
        self.n = n

    def _create_sql_query(self):
        return 'SELECT * FROM {} LIMIT {}'.format(self.sources[0].name, self.n)


##############################################################################
#                           Misc. API Functions
##############################################################################

def concat(objs):
    return Union(objs)

##############################################################################
#                           Criterion Classes
##############################################################################


class Equal(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('=', source_1, source_2, name=name)


class NotEqual(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<>', source_1, source_2, name=name)


class LessThan(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<', source_1, source_2, name=name)


class LessThanOrEqual(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<=', source_1, source_2, name=name)


class GreaterThan(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('>', source_1, source_2, name=name)


class GreaterThanOrEqual(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('>=', source_1, source_2, name=name)


class And(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('AND', source_1, source_2, name=name, simple=False)


class Or(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('OR', source_1, source_2, name=name, simple=False)


class Not(Criterion):
    def __init__(self, source, name=None):
        super().__init__('NOT', source, name=name, simple=False)

    def __str__(self):
        return 'NOT ({})'.format(self.sources[0])


##############################################################################
#                           Utility Functions
##############################################################################

def _define_dependencies(df: DataFrame):
    graph = _get_dependency_graph(df)
    ordered_deps = _topological_sort(graph)

    common_table_exprs = [
        '{} AS ({})'.format(t.name, t.sql(dependencies=False))
        for t in ordered_deps
        if not t.is_base_table and t is not df and t.result is None
    ]

    if len(common_table_exprs) > 0:
        return 'WITH ' + ', '.join(common_table_exprs)
    else:
        return None
