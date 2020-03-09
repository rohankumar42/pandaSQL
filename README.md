# pandaSQL

pandaSQL is a data-analysis library inspired by [pandas](https://github.com/pandas-dev/pandas), but designed to use existing database optimization techniques. While pandaSQL provides the familiar pandas-like API, internally, it uses SQLite to get  you results faster.


## Install

pandaSQL can be installed via `pip` as follows:
```bash
git clone https://github.com/rohankumar42/pandaSQL.git
cd pandaSQL
python3 -m pip install .
```


## How to Use

pandaSQL uses the same syntax that pandas does.

```python
> import pandasql as ps
> df = ps.read_csv('my_data.csv')    # or ps.DataFrame(pandas_df)
```

A crucial difference between pandaSQL and pandas is that pandaSQL is *lazy*. This means that when you say:
```python
> filtered = df[df['speed'] == 'fast']
```
`filtered` does not actually have any filtered results yet. Results are computed automatically when they are needed. For example, if you try to print `filtered`:

```python
> print(filtered)
       name speed
0  pandaSQL  fast
1   SQLite3  fast
```
The results are automatically computed for you.

## Development Note

pandaSQL is a fun project that [I](https://www.github.com/rohankumar42) have been working on in my spare time. If you run into any issues, let me know!
