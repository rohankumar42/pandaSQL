import time
import json
import argparse

import pandas
import dask
from dask import dataframe as dd
import pandasql


def order(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    if module is dd:
        books = dask.delayed(books)
    ordered = books.sort_values(by=["publication_year", "ISBN10"],
                                ascending=[False, False])
    oldest = ordered[["title", "publication_year", "ISBN10"]]

    if n is not None:
        oldest = oldest.head(n=n)

    if module is dd:
        oldest = dask.compute(oldest)[0]

    str(oldest)
    return oldest


def join_order(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    merged = module.merge(books, authors, on='first_name')
    merged["age"] = merged["publication_year"] - merged["birth_year"]

    if module is dd:
        merged = dask.delayed(merged)
    ordered = merged.sort_values(by=["age", "ISBN10"],
                                 ascending=[False, False])
    oldest = ordered[["title", "first_name",
                      "age", "ISBN10"]]

    if n is not None:
        oldest = oldest.head(n=n)
    if module is dd:
        oldest = dask.compute(oldest)[0]

    str(oldest)
    return oldest


def join_select(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    merged = module.merge(books, authors, on='first_name')
    merged["age"] = merged["publication_year"] - merged["birth_year"]

    if module is dd:
        merged = dask.delayed(merged)

    sel = merged[merged["age"] > 115]

    if n is not None:
        sel = sel.head(n=n)
    if module is dd:
        sel = dask.compute(sel)[0]

    str(sel)
    return sel


def big_join_select(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    authors = authors[['birth_day', 'birth_month',
                       'birth_year', 'bio', 'country']]
    if module is pandas:  # Suppress warning about writing to slice
        authors = pandas.DataFrame(authors)
    authors['dummy'] = 1
    books['dummy'] = 1

    merged = module.merge(books, authors, on='dummy')
    merged["age"] = merged["publication_year"] - merged["birth_year"]

    if module is dd:
        merged = dask.delayed(merged)

    sel = merged[merged["age"] > 115]

    if n is not None:
        sel = sel.head(n=n)
    if module is dd:
        sel = dask.compute(sel)[0]

    str(sel)
    return sel


def join(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    merged = module.merge(books, authors, on='first_name')

    if n is not None:
        merged = merged.head(n=n)

    str(merged)
    return merged


def triple_join(module, authors, books, top_authors, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))
    assert(isinstance(top_authors, module.DataFrame))

    first = module.merge(books, authors, on=['first_name', 'last_name'])
    merged = module.merge(first, top_authors, on=['first_name', 'last_name'])

    if n is not None:
        merged = merged.head(n=n)

    str(merged)
    return merged


def limit(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    head = books.head(n=n)

    str(head)
    return head


def selection(module, authors, books, n=None, **kwargs):
    assert(isinstance(authors, module.DataFrame))
    assert(isinstance(books, module.DataFrame))

    selection = books[(books['publication_year'] + 1 == 2020) |
                      (books['title'] == 'Scale Virtual Vortals')]

    if n is not None:
        selection = selection.head(n=n)
    str(selection)
    return selection


def run_benchmark(benchmark, nrows, limit=None):

    func = globals()[benchmark]
    top_authors_needed = benchmark == 'triple_join'
    print(f"Reading {nrows} CSV rows, running {benchmark} with limit={limit}")

    stats = {"pandas": {}, "pandaSQL": {}, "dask": {},
             "nrows": nrows, "benchmark": benchmark, "limit": limit}

    start = time.time()
    authors_df = pandas.read_csv('authors.csv')
    books_df = pandas.read_csv('books.csv', nrows=nrows)
    if top_authors_needed:
        top_authors_df = pandas.read_csv('top_authors.csv')
    else:
        top_authors_df = None
        top_authors = None
    time_taken = time.time() - start
    stats['pandas']['read_time'] = time_taken
    print("[Pandas]   Time taken to read: {:0.3f} seconds".format(time_taken))

    start = time.time()
    func(pandas, authors_df, books_df, n=limit, top_authors=top_authors_df)
    time_taken = time.time() - start
    stats['pandas']['run_time'] = time_taken
    print("[Pandas]   Time taken to run:  {:0.3f} seconds".format(time_taken))

    start = time.time()
    # authors = pandasql.read_csv('authors.csv')
    # books = pandasql.read_csv('books.csv')
    authors = pandasql.DataFrame(authors_df)
    books = pandasql.DataFrame(books_df)
    if top_authors_needed:
        top_authors = pandasql.DataFrame(top_authors_df)
    time_taken = time.time() - start
    stats['pandaSQL']['read_time'] = time_taken
    print("[PandaSQL] Time taken to read: {:0.3f} seconds".format(time_taken))

    start = time.time()
    func(pandasql, authors, books, n=limit, top_authors=top_authors)
    time_taken = time.time() - start
    stats['pandaSQL']['run_time'] = time_taken
    print("[PandaSQL] Time taken to run:  {:0.3f} seconds".format(time_taken))

    start = time.time()
    authors = dd.from_pandas(authors_df, npartitions=1)
    books = dd.from_pandas(books_df, npartitions=1)
    if top_authors_needed:
        top_authors = dd.from_pandas(top_authors_df, npartitions=1)
    time_taken = time.time() - start
    stats['dask']['read_time'] = time_taken
    print("[Dask]     Time taken to read: {:0.3f} seconds".format(time_taken))

    start = time.time()
    func(dd, authors, books, n=limit, top_authors=top_authors)
    time_taken = time.time() - start
    stats['dask']['run_time'] = time_taken
    print("[Dask]     Time taken to run:  {:0.3f} seconds".format(time_taken))

    print(json.dumps(stats, indent=4))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--nrows', type=int, default=10**3, required=True)
    parser.add_argument('--benchmark', type=str, required=True)
    parser.add_argument('--limit', type=int, default=None)
    args = parser.parse_args()

    run_benchmark(args.benchmark, args.nrows, args.limit)
