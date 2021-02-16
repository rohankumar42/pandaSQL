import pandasql as ps
import pandas as pd


def assertDataFrameEqualsPandas(df: ps.DataFrame,
                                expected_df: pd.DataFrame,
                                *args, **kwargs):
    result = df.compute()

    # Ignore Pandas index for the comparison
    result.reset_index(drop=True, inplace=True)
    expected_df.reset_index(drop=True, inplace=True)

    if isinstance(expected_df, pd.DataFrame):
        # Ignore order of columns for the comparison
        result = result[result.columns.sort_values()]
        expected_df = expected_df[expected_df.columns.sort_values()]

        pd.testing.assert_frame_equal(result, expected_df,
                                      *args, check_dtype=False, **kwargs)
    elif isinstance(expected_df, pd.Series):
        pd.testing.assert_series_equal(result, expected_df,
                                       *args, check_dtype=False, **kwargs)
    else:
        raise TypeError("Unexpected type {}".format(type(expected_df)))


def compute_all_ancestors(df: ps.DataFrame):
    for source in df.sources + df.pandas_sources:
        source._compute_pandas()
