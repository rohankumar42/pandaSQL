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

A = BaseTable(name='table1')
C = A['col']
print(C.sql())
D = A[A['col'] <= 10]
D2 = D[D['col'] > 5]
print(D2.sql())
D3 = D2['col']
print(D3.sql())
E = A[A['other'] != 'default']
print(D.sql())
F = A.join(D, on='col')
print(F.sql())
G = D.join(E, on='col')
print(G.sql())
