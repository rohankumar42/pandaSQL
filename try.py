from core import BaseTable, Equal, Constant, Projection, Selection

source = BaseTable(name='D')
ten = Constant(10)
proj1 = Projection(source, 'col1')
criterion = Equal(proj1, ten)
selection = Selection(source, criterion)
proj2 = Projection(selection, ['col1', 'col2'])
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

A = BaseTable(name='table1')
B = BaseTable(name='table2')
D = A[A['col'] <= 10]
try:
    D = B[A['col'] <= 10]
except ValueError as e:
    print(e)
E = D[['col1', 'col2']]
F = A[['col1', 'col2']]
G = A[A['col1'] == A['col2']]
print(D.sql())
print(E.sql())
print(F.sql())
print(G.sql())
