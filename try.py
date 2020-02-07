import pandas as pd
from core import BaseTable, Equal, Constant, Projection, Selection

# source = BaseTable(name='D')
# ten = Constant(10)
# proj1 = Projection(source, 'col1')
# criterion = Equal(proj1, ten)
# selection = Selection(source, criterion)
# proj2 = Projection(selection, ['col1', 'col2'])
# print(proj1.sql())
# print(selection.sql())
# print(proj2.sql())

"""
A = read_csv(...)           # ReadCSV(...)
B = A['col1']               # Projection(A, ['col1'])
C = B == 10                 # Equal(A, B)
D = A[C]                    # Selection(A, C)
E = D[['col1', 'col2']]     # Projection(D, ['col1', 'col2'])
"""

df = pd.DataFrame([{'num': i, 'val': str(i*100)} for i in range(10 ** 5)])
A = BaseTable.from_pandas(df, name='table1')
B = A['val']
print(B.sql())
# print(B.compute())
D = A[A['num'] <= 10]
D2 = D[D['num'] > 5]
print(D2.sql())
D3 = D2['num']
print(D3.sql())
E = A[A['val'] != '0']
print(D.sql())
F = A.join(D, on='num')
print(F.sql())
G = D.join(E, on='num')
print(G.sql())
# TODO: this fails because multiple "WITH {} AS {}" are apparently not valid
# SQL syntax. Probably need to precisely track dependencies, and before the
# main SQL query, run a topological sort to make one big common table
# expression section
print(G.compute())
