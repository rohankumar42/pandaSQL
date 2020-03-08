import time
import json
import argparse

import pandas
import pandasql


def offloading(min_n=3, max_n=6):

    times = []
    con = pandasql.core.SQL_CON

    for i in range(min_n, max_n + 1):
        nrows = 10 ** i
        start = time.time()
        _ = pandas.read_csv('books.csv', nrows=nrows)
        pandas_csv_time = time.time() - start

        start = time.time()
        df = pandasql.read_csv('books.csv', nrows=nrows)
        pandaSQL_csv_time = time.time() - start

        start = time.time()
        _ = pandas.read_sql(f'SELECT * FROM {df.name}', con=con)
        transfer_time = time.time() - start

        times.append({
            'nrows': nrows,
            'pandas_read_csv_time': pandas_csv_time,
            'pandaSQL_read_csv_time': pandaSQL_csv_time,
            'result_transfer_time': transfer_time
        })

    return times


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--min-n', type=int, default=3, required=True)
    parser.add_argument('--max-n', type=int, default=7, required=True)
    parser.add_argument('--outfile', type=str, default='overhead.json')
    args = parser.parse_args()

    times = offloading(args.min_n, args.max_n)
    with open(args.outfile, 'w') as fh:
        json.dump(times, fh, indent=4)
