SUPPORTED_VIA_SQLITE = {
    'all',
    'any',
    'columns',
    'count',
    'groupby',
    'head',
    'max',
    'mean',
    'memory_usage',
    'merge',
    'min',
    'prod',
    'rename',
    'sort_values',
    'sum',
    'to_csv',
    'to_json',
    'to_numpy',
    'to_pickle',
    'update'
}

SUPPORTED_VIA_FALLBACK = {
    'nlargest',
    'nsmallest',
    # TODO: try to add all unsupported pandas ops
}

UNSUPPORTED = {
    'T',
    'abs',
    'add',
    'add_prefix',
    'add_suffix',
    'agg',
    'aggregate',
    'align',
    'append',
    'apply',
    'applymap',
    'asfreq',
    'asof',
    'assign',
    'astype',
    'at',
    'at_time',
    'attrs',
    'axes',
    'between_time',
    'bfill',
    'bool',
    'boxplot',
    'clip',
    'combine',
    'combine_first',
    'convert_dtypes',
    'copy',
    'corr',
    'corrwith',
    'cov',
    'cummax',
    'cummin',
    'cumprod',
    'cumsum',
    'describe',
    'diff',
    'div',
    'divide',
    'dot',
    'drop',
    'drop_duplicates',
    'droplevel',
    'dropna',
    'dtypes',
    'duplicated',
    'empty',
    'eq',
    'equals',
    'eval',
    'ewm',
    'expanding',
    'explode',
    'ffill',
    'fillna',
    'filter',
    'first',
    'first_valid_index',
    'floordiv',
    'from_dict',
    'from_records',
    'ge',
    'get',
    'gt',
    'hist',
    'iat',
    'idxmax',
    'idxmin',
    'iloc',
    'index',
    'infer_objects',
    'info',
    'insert',
    'interpolate',
    'isin',
    'isna',
    'isnull',
    'items',
    'iteritems',
    'iterrows',
    'itertuples',
    'join',
    'keys',
    'kurt',
    'kurtosis',
    'last',
    'last_valid_index',
    'le',
    'loc',
    'lookup',
    'lt',
    'mad',
    'mask',
    'median',
    'melt',
    'mod',
    'mode',
    'mul',
    'multiply',
    'ndim',
    'ne',
    'nlargest',
    'notna',
    'notnull',
    'nsmallest',
    'nunique',
    'pct_change',
    'pipe',
    'pivot',
    'pivot_table',
    'plot',
    'pop',
    'pow',
    'product',
    'quantile',
    'query',
    'radd',
    'rank',
    'rdiv',
    'reindex',
    'reindex_like',
    'rename_axis',
    'reorder_levels',
    'replace',
    'resample',
    'reset_index',
    'rfloordiv',
    'rmod',
    'rmul',
    'rolling',
    'round',
    'rpow',
    'rsub',
    'rtruediv',
    'sample',
    'select_dtypes',
    'sem',
    'set_axis',
    'set_index',
    'shape',
    'shift',
    'size',
    'skew',
    'slice_shift',
    'sort_index',
    'squeeze',
    'stack',
    'std',
    'style',
    'sub',
    'subtract',
    'swapaxes',
    'swaplevel',
    'tail',
    'take',
    'to_clipboard',
    'to_dict',
    'to_excel',
    'to_feather',
    'to_gbq',
    'to_hdf',
    'to_html',
    'to_latex',
    'to_markdown',
    'to_parquet',
    'to_period',
    'to_records',
    'to_sql',
    'to_stata',
    'to_string',
    'to_timestamp',
    'to_xarray',
    'transform',
    'transpose',
    'truediv',
    'truncate',
    'tshift',
    'tz_convert',
    'tz_localize',
    'unstack',
    'values',
    'var',
    'where',
    'xs'
}