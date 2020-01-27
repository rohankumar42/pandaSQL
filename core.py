import uuid

# TODO: add more supported types
SUPPORTED_TYPES = [int, float, str]


def _is_supported_constant(x):
    return any(isinstance(x, t) for t in SUPPORTED_TYPES)


# TODO: maybe think of a better name than thunk
class BaseThunk(object):
    def __init__(self, name=None):
        self.name = name or uuid.uuid4().hex

    def sql(self):
        raise TypeError('Objects of type {} cannot be converted to a SQL query'
                        .format(type(self)))

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


class BaseTable(BaseThunk):
    # TODO: cache computed results
    def __init__(self, name=None):
        super().__init__(name=name)
        self.base_table = self

    @property
    def is_base_table(self):
        return self.base_table is self

    def __getitem__(self, x):
        if isinstance(x, str) or isinstance(x, list):  # TODO: check valid cols
            return Projection(self, x)
        elif isinstance(x, BaseCriterion):
            return Selection(self, x)
        elif isinstance(x, int):
            raise NotImplementedError('TODO: iloc/loc based access')
        else:
            raise TypeError('Unsupported indexing type {}'.format(type(x)))

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
        self.source = source
        self.base_table = self.source.base_table
        self.cols = cols

    def __str__(self):
        attrs = ', '.join('{}.{}'.format(self.source, col)
                          for col in self.cols)
        if len(self.cols) > 1:
            attrs = '({})'.format(attrs)
        return attrs

    def sql(self):
        if self.source.is_base_table:
            source_sql = self.source.name
        else:
            source_sql = '({})'.format(self.source.sql())

        return 'SELECT {} FROM {}'.format(', '.join(self.cols), source_sql)


class Selection(BaseTable):
    def __init__(self, source, criterion, name=None):
        assert(isinstance(source, BaseTable))
        assert(isinstance(criterion, BaseCriterion))

        sources = criterion.sources
        if len(sources) != 0 and source.name not in sources:
            raise ValueError('Cannot select from table {} with a '
                             'criterion for table(s) {}'
                             .format(source.name, sources))

        super().__init__(name=name)
        self.source = source
        self.base_table = self.source.base_table
        self.criterion = criterion

    def __str__(self):
        return 'Select({}, {})'.format(self.source, self.criterion)

    def sql(self):
        if self.source.is_base_table:
            source_sql = self.source.name
        else:
            source_sql = '({})'.format(self.source.sql())

        return 'SELECT * FROM {} WHERE {}'.format(source_sql, self.criterion)


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
        self.source_1 = source_1
        self.source_2 = source_2
        self.sources = set()
        if isinstance(self.source_1, Projection):
            self.sources.add(self.source_1.base_table.name)
        if isinstance(self.source_2, Projection):
            self.sources.add(self.source_2.base_table.name)

    def __str__(self):
        return '{} {} {}'.format(self.source_1, self.operation, self.source_2)


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
        raise NotImplementedError
