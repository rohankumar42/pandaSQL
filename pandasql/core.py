# import uuid
import pandas as pd
from typing import List

from pandasql.utils import _is_supported_constant, _get_dependency_graph, \
    _topological_sort, _new_name
from pandasql.sql_utils import get_sqlite_connection

SQL_CON = get_sqlite_connection()


class BaseThunk(object):
    def __init__(self, name=None):
        self.name = name or _new_name()
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
        if isinstance(self.value, str):
            return "'{}'".format(self.value)
        else:
            return str(self.value)


class Criterion(BaseThunk):
    def __init__(self, operation, source_1, source_2=None,
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


class DataFrame(BaseThunk):
    def __init__(self, data=None, name=None, sources=None, base_tables=None):
        super().__init__(name=name)
        # TODO: deduplicate sources and base_tables
        self.sources = sources or []
        self.base_tables = list(base_tables or [self])
        self.dependents = []
        self.update = None
        self.result = None
        self.columns = None
        df = None

        # For each source, add this as a dependent
        for source in self.sources:
            source.dependents.append(self)

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

        if df is not None and len(df) > 0:
            # Offload dataframe to SQLite
            df.to_sql(name=self.name, con=SQL_CON, index=False)

            # Store columns
            self.columns = df.columns

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
            # Compute result and store in SQLite table
            query = self.sql(dependencies=True)
            compute_query = 'CREATE TABLE {} AS {}'.format(self.name, query)
            SQL_CON.execute(compute_query)

            # Read table as Pandas DataFrame
            read_query = 'SELECT * FROM {}'.format(self.name)
            self.result = pd.read_sql_query(read_query, con=SQL_CON)
            self.columns = self.result.columns

        return self.result

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
        new = DataFrame(sources=[old], base_tables=old.base_tables)
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

        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)

        if len(pd.Index(cols).difference(source.columns)) > 0:
            raise ValueError("Projection columns {} are not a subset of {}"
                             .format(cols, source.columns))
        self.columns = source.columns[source.columns.isin(cols)]

        self._sql_query = 'SELECT {} FROM {}'.format(', '.join(self.columns),
                                                     self.sources[0].name)

    def __hash__(self):
        return super().__hash__()


class Selection(DataFrame):
    def __init__(self, source: DataFrame, criterion: Criterion, name=None):
        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)
        self.criterion = criterion
        self.columns = source.columns

        self._sql_query = 'SELECT * FROM {} WHERE {}'.format(
            self.sources[0].name, self.criterion)


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
        super().__init__(name=name, sources=[source_1, source_2],
                         base_tables=source_1.base_tables +
                         source_2.base_tables)
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
        base_tables = list({base for s in sources for base in s.base_tables})
        super().__init__(name=name, sources=sources, base_tables=base_tables)

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

        super().__init__(name=name, sources=[source],
                         base_tables=source.base_tables)
        self.n = n
        self.columns = source.columns

        self._sql_query = 'SELECT * FROM {} LIMIT {}'.format(
            self.sources[0].name, self.n)


##############################################################################
#                           Misc. API Functions
##############################################################################


def merge(left, right, on=None):
    """TODO: support other pandas join arguments"""
    return Join(left, right, on)


def concat(objs: List[DataFrame]):
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
#                           Arithmetic Classes
##############################################################################


class Arithmetic(DataFrame, ArithmeticOperand):
    def __init__(self, operation, operand_1, operand_2=None, inline=False,
                 name=None):

        base_tables = []
        sources = []

        self.operand_1 = _make_projection_or_constant(operand_1, simple=True)
        if isinstance(self.operand_1, DataFrame):
            base_tables += self.operand_1.base_tables
            sources += self.operand_1.sources

        if operand_2 is not None:
            self.operand_2 = _make_projection_or_constant(operand_2,
                                                          simple=True)
            if isinstance(self.operand_2, DataFrame):
                base_tables += self.operand_2.base_tables
                sources += self.operand_2.sources

        self.unary = operand_2 is None

        super().__init__(name=name, sources=sources, base_tables=base_tables)

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
#                           Utility Functions
##############################################################################


def _define_dependencies(df: DataFrame):
    graph = _get_dependency_graph(df)
    ordered_deps = _topological_sort(graph)

    common_table_exprs = [
        '{} AS ({})'.format(t.name, t.sql(dependencies=False))
        for t in ordered_deps
        if t is not df and t.result is None
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
