import pandas as pd
from typing import List

from pandasql.utils import _is_supported_constant, _get_dependency_graph, \
    _topological_sort, _new_name
from pandasql.sql_utils import get_sqlite_connection


SQL_CON = get_sqlite_connection()
OFFLOADING_STRATEGY = None


def require_result(func):
    '''Decorator for functions that require results to be ready'''

    def result_ensured(self, *args, **kwargs):
        self.compute()
        return func(self, *args, **kwargs)

    return result_ensured


class BaseFrame(object):
    def __init__(self, name=None, sources=None):
        self.name = name or _new_name()
        # TODO: deduplicate sources
        self.sources = sources or []
        self.dependents = []
        self._cached_result = None
        self.update = None
        self.columns = pd.Index([])
        self._sql_query = None

        # For each source, add this as a dependent
        for source in self.sources:
            source.dependents.append(self)

    @property
    def result(self):
        if self._cached_result is None:
            return None
        elif hasattr(self, 'post_process_result'):
            return self.post_process_result(self._cached_result)
        else:
            return self._cached_result

    def compute(self):

        if self._cached_result is None:
            if should_offload_computation(self):
                self._compute_sqlite()
            else:
                self._compute_pandas()

        return self.result

    def _compute_pandas(self):
        if self._cached_result is None:
            graph = _get_dependency_graph(self)
            ordered_deps = _topological_sort(graph)

            # Compute all dependencies in order (via Pandas)
            # TODO: should GroupByDataFrame be excluded here too?
            for t in ordered_deps:
                if t is not self and t.result is None \
                        and isinstance(t, BaseFrame):
                    t._compute_pandas()

            # Finally, compute this object's result
            # TODO(important): when should this result be offloaded to SQLite?
            self._cached_result = self._pandas()

        return self.result

    def _compute_sqlite(self):
        if self._cached_result is None:
            # Compute result and store in SQLite table
            query = self.sql(dependencies=True)
            compute_query = 'CREATE TABLE {} AS {}'.format(self.name, query)
            SQL_CON.execute(compute_query)

            # Read table as Pandas DataFrame
            read_query = 'SELECT * FROM {}'.format(self.name)
            self._cached_result = pd.read_sql_query(read_query, con=SQL_CON)
            self.columns = self._cached_result.columns

        return self.result

    def _pandas(self):
        raise NotImplementedError("To be implemented by subclasses")

    def sql(self, dependencies=True):
        query = []

        if dependencies:
            common_table_expr = _define_dependencies(self)
            if common_table_expr is not None:
                query.append(common_table_expr)

        if self.update is None:
            query.append(self._sql_query)
        else:
            query.append(self.update._sql_query)

        return ' '.join(query)

    def sum(self):
        return Aggregator('SUM', self)

    def mean(self):
        return Aggregator('AVG', self)

    def count(self):
        return Aggregator('COUNT', self)

    def min(self):
        return Aggregator('MIN', self)

    def max(self):
        return Aggregator('MAX', self)

    def prod(self):
        return Aggregator('PROD', self)

    def any(self):
        return Aggregator('AGG_ANY', self)

    def all(self):
        return Aggregator('AGG_ALL', self)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)

    def __hash__(self):
        return hash(self.name)


class Constant(BaseFrame):
    def __init__(self, value, name=None):
        super().__init__(name=name)
        if not _is_supported_constant(value):
            raise TypeError('Unsupported type {}'.format(type(value)))
        self._cached_result = value

    def __str__(self):
        if isinstance(self._cached_result, str):
            return "'{}'".format(self._cached_result)
        else:
            return str(self._cached_result)


class Criterion(BaseFrame):
    def __init__(self, operation, pandas_func, source_1, source_2=None,
                 name=None, simple=True):
        # Simple criteria depend on comparisons between columns and/or
        # constants, whereas compound criteria depend on smaller criteria
        if simple:
            assert(isinstance(source_1, ArithmeticOperand) or
                   isinstance(source_1, Constant))
            if source_2 is not None:
                assert(isinstance(source_2, ArithmeticOperand) or
                       isinstance(source_2, Constant))
        else:
            assert(isinstance(source_1, Criterion))
            if source_2 is not None:
                assert(isinstance(source_2, Criterion))

        assert(callable(pandas_func))
        self._pandas_func = pandas_func

        self.operation = operation
        sources = [source_1]
        if source_2 is not None:
            sources.append(source_2)

        super().__init__(name=name, sources=sources)

    # TODO: put this in self.as_sql_criterion() and make __str__()
    # actually compute like other BaseFrames
    def __str__(self):
        '''Default for binary Criterion objects'''
        return '{} {} {}'.format(self._source_to_str(self.sources[0]),
                                 self.operation,
                                 self._source_to_str(self.sources[1]))

    def _pandas(self):
        results = [s.result[s.columns[0]] if isinstance(s, Projection)
                   else s.result for s in self.sources]

        return self._pandas_func(*results)

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
                              for col in source.columns)
            if len(source.columns) > 1:
                attrs = '({})'.format(attrs)
            return attrs
        elif isinstance(source, Arithmetic):
            return source._operation_as_str()
        elif isinstance(source, Constant) or isinstance(source, Criterion):
            return str(source)
        else:
            raise TypeError("Unexpected source type {}".format(type(source)))


class ArithmeticOperand(object):
    def __add__(self, other):
        return Add(self, other)

    def __sub__(self, other):
        return Subtract(self, other)

    def __mul__(self, other):
        return Multiply(self, other)

    def __truediv__(self, other):
        return Divide(self, other)

    def __floordiv__(self, other):
        return FloorDivide(self, other)

    def __mod__(self, other):
        return Modulo(self, other)

    def __pow__(self, other):
        return Power(self, other)

    def __and__(self, other):
        return BitAnd(self, other)

    def __or__(self, other):
        return BitOr(self, other)

    def __xor__(self, other):
        return BitXor(self, other)

    def __radd__(self, other):
        return Add(other, self)

    def __rsub__(self, other):
        return Subtract(other, self)

    def __rmul__(self, other):
        return Multiply(other, self)

    def __rtruediv__(self, other):
        return Divide(other, self)

    def __rfloordiv__(self, other):
        return FloorDivide(other, self)

    def __rmod__(self, other):
        return Modulo(other, self)

    def __rpow__(self, other):
        return Power(other, self)

    def __rand__(self, other):
        return BitAnd(other, self)

    def __ror__(self, other):
        return BitOr(other, self)

    def __rxor__(self, other):
        return BitXor(other, self)

    def __invert__(self):
        return Invert(self)

    def __neg__(self):
        return Multiply(self, -1)

    def __abs__(self):
        return Abs(self)

    def __eq__(self, other):
        return self.__comparison(other, Equal)

    def __ne__(self, other):
        return self.__comparison(other, NotEqual)

    def __lt__(self, other):
        return self.__comparison(other, LessThan)

    def __le__(self, other):
        return self.__comparison(other, LessThanOrEqual)

    def __gt__(self, other):
        return self.__comparison(other, GreaterThan)

    def __ge__(self, other):
        return self.__comparison(other, GreaterThanOrEqual)

    def __comparison(self, other, how_class):
        # TODO: move the _make call into Criterion.__init__
        return how_class(self, _make_projection_or_constant(other))


class DataFrame(BaseFrame):
    def __init__(self, data=None, name=None, sources=None):
        super().__init__(name=name, sources=sources)
        df = None

        # If data provided, result is already ready
        if isinstance(data, dict) or isinstance(data, list):
            df = pd.DataFrame(data)
            self._cached_result = df
        elif isinstance(data, pd.DataFrame):
            df = data
            self._cached_result = df
        elif isinstance(data, DataFrame):
            df = data.compute()
            self._cached_result = df
        elif data is None:
            pass
        else:
            raise TypeError('Cannot create table from object of type {}'
                            .format(type(data)))

        if df is not None and len(df) > 0:
            # Offload dataframe to SQLite
            df.to_sql(name=self.name, con=SQL_CON, index=False)

            # Store columns
            self.columns = df.columns

    def __getitem__(self, x):
        if isinstance(x, str) or isinstance(x, list):
            return Projection(self, x)
        elif isinstance(x, Criterion):
            return Selection(self, x)
        elif isinstance(x, slice):
            if x.start is not None or x.step is not None:
                raise ValueError('Only slices of the form df[:n] are accepted')
            return self if x.stop is None else Limit(self, n=x.stop)
        else:
            raise TypeError('Unsupported indexing type {}'.format(type(x)))

    def __setitem__(self, col, value):
        # Make a copy of self which will act as the source of all DataFrames
        # which already depend on self
        old = DataFrame()
        old.__dict__.update(self.__dict__)

        # For all dependents, replace self as a source, and add old instead
        for dependent in old.dependents:
            dependent.sources.remove(self)
            dependent.sources.append(old)

        # Create a new DataFrame (which will become self), and add old
        # as a source, tracking the update that is being done
        new = DataFrame(sources=[old])
        new.update = Update(old, new, col, value)

        # If column is new, add it to columns
        new.columns = old.columns
        if col not in old.columns:
            new.columns = old.columns.insert(len(old.columns), col)

        # Update self to new object
        self.__dict__.update(new.__dict__)

    def head(self, n=5):
        return self[:n]

    def sort_values(self, by, ascending=True):
        return OrderBy(self, cols=by, ascending=ascending)

    def merge(self, other, on=None):
        """TODO: support other pandas join arguments"""
        if on is None:
            raise NotImplementedError('Joins without key(s) are not supported')
        else:
            return Join(self, other, on)

    def groupby(self, by, as_index=False):
        """TODO: support other pandas groupby arguments"""
        return GroupByDataFrame(self, by=by, as_index=as_index)

    @require_result
    def __str__(self):
        return str(self.result)

    @require_result
    def __len__(self):
        return len(self.result)

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

    def __hash__(self):
        return super().__hash__()


class Update(object):
    def __init__(self, source: DataFrame, dest: DataFrame, col: str, value):
        # TODO: check if projection is from the same df?

        self.source = source
        self.dest = dest
        self.col = col
        self.value = _make_projection_or_constant(value, simple=True)

        columns = self.source.columns
        columns = columns.drop(self.col) if self.col in columns else columns

        val = self.value
        if isinstance(val, Projection):
            val = val.columns[0]
        elif isinstance(val, Arithmetic):
            val = val._operation_as_str()

        new_column = '{} AS {}'.format(val, self.col)
        columns = columns.insert(len(columns), new_column)

        self._sql_query = 'SELECT {} FROM {}'.format(', '.join(columns),
                                                     self.source.name)

    def __str__(self):
        val = self.value
        if isinstance(val, Projection):
            val = val.columns[0]
        return 'Update({} to {}, {} <- {})'.format(self.source.name,
                                                   self.dest.name, self.col,
                                                   val)


class Projection(DataFrame, ArithmeticOperand):
    def __init__(self, source: DataFrame, col, name=None):

        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, list):
            cols = col
        else:
            raise TypeError('col must be of type str or list, but found {}'
                            .format(type(col)))

        super().__init__(name=name, sources=[source])

        if len(pd.Index(cols).difference(source.columns)) > 0:
            raise ValueError("Projection columns {} are not a subset of {}"
                             .format(cols, source.columns))
        self.columns = source.columns[source.columns.isin(cols)]

        self._sql_query = 'SELECT {} FROM {}'.format(', '.join(self.columns),
                                                     self.sources[0].name)

    def _pandas(self):
        return self.sources[0].result[self.columns]

    def __hash__(self):
        return super().__hash__()


class Selection(DataFrame):
    def __init__(self, source: DataFrame, criterion: Criterion, name=None):
        super().__init__(name=name, sources=[source])
        self.criterion = criterion
        self.columns = source.columns

        self._sql_query = 'SELECT * FROM {} WHERE {}'.format(
            self.sources[0].name, self.criterion)

    def _pandas(self):
        # self.criterion might not already be computed since it is not
        # technically a "source" (dependency). So, explicitly compute it.
        return self.sources[0].result[self.criterion.compute()]


class OrderBy(DataFrame):
    def __init__(self, source: DataFrame, cols, ascending=True, name=None):

        super().__init__(name=name, sources=[source])
        if isinstance(cols, str):
            cols = [cols]
        if isinstance(ascending, bool):
            ascending = [ascending]
        if len(cols) != len(ascending):
            raise ValueError("cols and ascending must be equal lengths, "
                             "but found len(cols)={} and len(ascending)={}"
                             .format(len(cols), len(ascending)))

        self.order_cols = cols
        self.ascending = ascending
        self.columns = source.columns

        order_by = [
            '{}.{} {}'.format(self.sources[0].name, col,
                              'ASC' if asc else 'DESC')
            for col, asc in zip(self.order_cols, self.ascending)
        ]

        self._sql_query = 'SELECT * FROM {} ORDER BY {}' \
            .format(self.sources[0].name, ', '.join(order_by))


class Join(DataFrame):
    def __init__(self, source_1: DataFrame, source_2: DataFrame,
                 join_keys, name=None):
        super().__init__(name=name, sources=[source_1, source_2])
        if isinstance(join_keys, str):
            join_keys = [join_keys]
        self.join_keys = join_keys

        join_index = pd.Index(join_keys)
        for source in self.sources:
            if len(join_index.difference(source.columns)) > 0:
                raise ValueError("Source {} does not contain all join keys {}"
                                 .format(source.name, join_keys))

        self.columns = source_1.columns.append(
            source_2.columns.drop(join_keys))

        self._sql_query = 'SELECT * FROM {} JOIN {} USING ({})' \
            .format(self.sources[0].name, self.sources[1].name,
                    ','.join(self.join_keys))


class Union(DataFrame):
    def __init__(self, sources: List[DataFrame], name=None):
        super().__init__(name=name, sources=sources)

        self.columns = pd.Index([])
        schema = sources[0].columns
        for source in sources:
            self.columns = self.columns.append(source.columns)
            if len(source.columns.symmetric_difference(schema)) > 0:
                raise ValueError("Cannot union sources with different schemas")

        self._sql_query = ' UNION ALL '.join('SELECT * FROM {}'
                                             .format(source.name)
                                             for source in self.sources)


class Limit(DataFrame):
    def __init__(self, source: DataFrame, n: int, name=None):

        super().__init__(name=name, sources=[source])
        self.n = n
        self.columns = source.columns

        self._sql_query = 'SELECT * FROM {} LIMIT {}'.format(
            self.sources[0].name, self.n)


##############################################################################
#                           GroupBy Operations
##############################################################################

class GroupByDataFrame(BaseFrame):
    def __init__(self, source: DataFrame, by, as_index=False, name=None):

        if isinstance(by, str):
            self.groupby_cols = [by]
        elif isinstance(by, list):
            self.groupby_cols = by
        else:
            raise TypeError('by must be of type str or list, but found {}'
                            .format(type(by)))

        super().__init__(name=name, sources=[source])
        self.base_name = source.name
        self.columns = source.columns
        self.as_index = as_index

    def __getitem__(self, x):
        if isinstance(x, str) or isinstance(x, list):
            return GroupByProjection(self, x)
        elif isinstance(x, slice):
            raise TypeError('Slicing not support for GroupBy objects')
        else:
            raise TypeError('Unsupported indexing type {}'.format(type(x)))

    def __str__(self):
        return 'GroupBy({}, by={})'.format(self.sources[0].name,
                                           self.groupby_cols)


class GroupByProjection(GroupByDataFrame):
    def __init__(self, source: GroupByDataFrame, col, name=None):

        if isinstance(col, str):
            cols = [col]
        elif isinstance(col, list):
            cols = col
        else:
            raise TypeError('col must be of type str or list, but found {}'
                            .format(type(col)))

        super().__init__(source.sources[0], by=source.groupby_cols,
                         as_index=source.as_index, name=name)

        if len(pd.Index(cols).difference(source.columns)) > 0:
            raise ValueError("Projection columns {} are not a subset of {}"
                             .format(cols, source.columns))
        self.columns = source.columns[source.columns.isin(cols)]

    def __str__(self):
        return 'GroupByProjection({}, by={}, cols={})' \
            .format(self.sources[0].name, self.groupby_cols,
                    self.columns.to_list())


##############################################################################
#                                Aggregators
##############################################################################


class Aggregator(DataFrame):
    VALID_AGGREGATORS = ['SUM', 'COUNT', 'AVG', 'MIN', 'MAX',
                         'PROD', 'AGG_ANY', 'AGG_ALL']

    def __init__(self, agg, source: BaseFrame, name=None):
        super().__init__(sources=[source], name=name)

        assert(agg in self.VALID_AGGREGATORS)
        self.agg = agg
        self.grouped = isinstance(source, GroupByDataFrame)

        # TODO: only use valid columns for aggregation operation
        cols = ['{}({}) AS {}'.format(agg, c, c)
                for c in source.columns]

        if not self.grouped:
            self._sql_query = 'SELECT {} FROM {}'.format(', '.join(cols),
                                                         source.name)
        else:
            cols = list(source.groupby_cols)
            cols += ['{}({}) AS {}'.format(agg, c, c)
                     for c in source.columns
                     if c not in source.groupby_cols]

            self._sql_query = 'SELECT {} FROM {} GROUP BY {}' \
                .format(', '.join(cols), source.base_name,
                        ', '.join(source.groupby_cols))

        self.columns = source.columns

        # Set return types in cases where explicit conversion is needed,
        # e.g., bool because SQLite doesn't have bools
        if agg in ['AGG_ANY', 'AGG_ALL']:
            self.final_type = bool
        else:
            self.final_type = None

    def post_process_result(self, result):
        '''This function will be called by BaseFrame.compute'''
        if self.grouped and self.sources[0].as_index:
            # Add index to the computed Pandas DataFrame, like Pandas would
            ret = result.set_index(self.sources[0].groupby_cols)
        elif len(result) == 1:
            series = result.iloc[0]
            if len(series) == 1:    # Single numerical value
                ret = series[0]
            else:                   # Multiple numerical values
                series.name = None
                ret = series
        else:
            ret = result

        # Type cast if necessary
        if self.final_type is not None:
            if isinstance(ret, pd.DataFrame) or isinstance(ret, pd.Series):
                ret = ret.astype(self.final_type)
            else:
                ret = self.final_type(ret)

        return ret


##############################################################################
#                           Misc. API Functions
##############################################################################


def merge(left: DataFrame, right: DataFrame, on=None):
    """TODO: support other pandas join arguments"""
    return Join(left, right, on)


def concat(objs: List[DataFrame]):
    return Union(objs)


##############################################################################
#                           Criterion Classes
##############################################################################


class Equal(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('=', lambda x, y: x == y,
                         source_1, source_2, name=name)


class NotEqual(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<>', lambda x, y: x != y,
                         source_1, source_2, name=name)


class LessThan(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<', lambda x, y: x < y,
                         source_1, source_2, name=name)


class LessThanOrEqual(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('<=', lambda x, y: x <= y,
                         source_1, source_2, name=name)


class GreaterThan(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('>', lambda x, y: x > y,
                         source_1, source_2, name=name)


class GreaterThanOrEqual(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('>=', lambda x, y: x >= y,
                         source_1, source_2, name=name)


class And(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('AND', lambda x, y: x & y,
                         source_1, source_2, name=name, simple=False)


class Or(Criterion):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('OR', lambda x, y: x | y,
                         source_1, source_2, name=name, simple=False)


class Not(Criterion):
    def __init__(self, source, name=None):
        super().__init__('NOT', lambda x: ~x,
                         source, name=name, simple=False)

    def __str__(self):
        return 'NOT ({})'.format(self.sources[0])

##############################################################################
#                           Arithmetic Classes
##############################################################################


class Arithmetic(DataFrame, ArithmeticOperand):
    def __init__(self, operation, operand_1, operand_2=None, inline=False,
                 name=None):

        sources = []

        self.operand_1 = _make_projection_or_constant(operand_1, simple=True)
        if isinstance(self.operand_1, DataFrame):
            sources += self.operand_1.sources

        if operand_2 is not None:
            self.operand_2 = _make_projection_or_constant(operand_2,
                                                          simple=True)
            if isinstance(self.operand_2, DataFrame):
                sources += self.operand_2.sources

        self.unary = operand_2 is None

        super().__init__(name=name, sources=sources)

        self.operation = operation
        self.inline = inline

        self._sql_query = 'SELECT {} AS res FROM {}'.format(
            self._operation_as_str(), self.sources[0].name)

    def _operation_as_str(self):
        if self.unary:
            fmt = '{op}{x}' if self.inline else'{op}({x})'

            return fmt.format(x=self._operand_to_str(self.operand_1),
                              op=self.operation)
        else:
            fmt = '{x} {op} {y}' if self.inline else'{op}({x}, {y})'

            return fmt.format(x=self._operand_to_str(self.operand_1),
                              y=self._operand_to_str(self.operand_2),
                              op=self.operation)

    @staticmethod
    def _operand_to_str(source):
        if isinstance(source, Projection):
            return '{}.{}'.format(source.sources[0].name, source.columns[0])
        elif isinstance(source, Constant):
            return str(source)
        elif isinstance(source, Arithmetic):
            if source.inline:
                return '({})'.format(source._operation_as_str())
            else:
                return source._operation_as_str()
        else:
            raise TypeError("Unexpected source type {}".format(type(source)))

    def _pandas(self):
        raise NotImplementedError


class Add(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('+', source_1, source_2, name=name, inline=True)


class Subtract(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('-', source_1, source_2, name=name, inline=True)


class Multiply(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('*', source_1, source_2, name=name, inline=True)


class Divide(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('DIV', source_1, source_2, name=name)


class FloorDivide(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('FLOORDIV', source_1, source_2, name=name)


class Modulo(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('MOD', source_1, source_2, name=name)


class Power(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('POW', source_1, source_2, name=name)


class BitAnd(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('BITAND', source_1, source_2, name=name)


class BitOr(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('BITOR', source_1, source_2, name=name)


class BitXor(Arithmetic):
    def __init__(self, source_1, source_2, name=None):
        super().__init__('BITXOR', source_1, source_2, name=name)


class Invert(Arithmetic):
    def __init__(self, source, name=None):
        super().__init__('INV', source, name=name)


class Abs(Arithmetic):
    def __init__(self, source, name=None):
        super().__init__('abs', source, name=name)


##############################################################################
#                       Offloading Decision Functions
##############################################################################


def offloading_strategy(name=None):
    if name is not None:
        name = name.upper()
        if name not in ['ALWAYS', 'NEVER']:
            raise ValueError(f'Unsupported offloading strategy: {name}')

        global OFFLOADING_STRATEGY
        OFFLOADING_STRATEGY = name

    else:
        return OFFLOADING_STRATEGY


def should_offload_computation(df: BaseFrame):
    if OFFLOADING_STRATEGY == 'ALWAYS':
        return True
    elif OFFLOADING_STRATEGY == 'NEVER':
        return False
    else:
        # TODO: put smart offloading logic here
        raise NotImplementedError('Unsupported offloading strategy: {}'
                                  .format(OFFLOADING_STRATEGY))


##############################################################################
#                           Utility Functions
##############################################################################


def _define_dependencies(df: DataFrame):
    graph = _get_dependency_graph(df)
    ordered_deps = _topological_sort(graph)

    # Do NOT define a dependency t if any of the following is true:
    #   (1) t is the current df
    #   (2) t.result is not None (result cached)
    #   (3) t is a GroupByDataFrame object
    common_table_exprs = [
        '{} AS ({})'.format(t.name, t.sql(dependencies=False))
        for t in ordered_deps
        if t is not df and t.result is None and isinstance(t, DataFrame)
    ]

    if len(common_table_exprs) > 0:
        return 'WITH ' + ', '.join(common_table_exprs)
    else:
        return None


def _make_projection_or_constant(x, simple=False, arithmetic=True):
    if isinstance(x, Projection):
        if simple and len(x.columns) != 1:
            raise ValueError("Projections must have exactly 1 column"
                             "but found columns {}".format(x.columns))
        return x
    elif _is_supported_constant(x):
        return Constant(x)
    elif arithmetic and isinstance(x, Arithmetic):
        return x
    else:
        raise TypeError('Only constants and Projections are accepted')


def stop():
    SQL_CON.close()
