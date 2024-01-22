"""
Esercitazione: 2
Esercizio: 1.1
Autore: Alessandro Amella
Matricola: 0001070569
"""

"""1. matrici e norme """

import numpy as np

n = 2
A = np.array([[1, 2], [0.499, 1.001]])

print ('Norme di A:')
norm1 = np.linalg.norm(A, 1) 
norm2 = np.linalg.norm(A, 2) 
normfro = np.linalg.norm(A, 'fro') 
norminf = np.linalg.norm(A, np.inf)

print('Norma1 = ', norm1, '\n') # 3.001
print('Norma2 = ', norm2, '\n') # 2.500200104037774
print('Normafro = ', normfro, '\n') # 2.5002003919686118
print('Norma infinito = ', norminf, '\n') # 3.0

cond1 = np.linalg.cond(A, 1)
cond2 = np.linalg.cond(A, 2)
condfro = np.linalg.cond(A, 'fro')
condinf = np.linalg.cond(A, np.inf)

print ('K(A)_1 = ', cond1, '\n') # 3001.0000000001082
print ('K(A)_2 = ', cond2, '\n') # 2083.666853410337
print ('K(A)_fro =', condfro, '\n') # 2083.6673333334084
print ('K(A)_inf =', condinf, '\n') # 3001.0000000001082

x = np.ones((2,1))
b = A @ x
print ('b = ', b) # [[3.   ], [1.5]]

btilde = np.array([[3], [1.4985]])
xtilde = np.array([[2, 0.5]]).T

# Verificare che xtilde è soluzione di A xtilde = btilde (Axtilde)
my_btilde = A @ xtilde

print ('A*xtilde = ', btilde) # [[3.    ], [1.4985]]
print(np.linalg.norm(btilde-my_btilde,'fro')) # 0.0

deltax = np.linalg.norm(x-xtilde, ord=2)
deltab = np.linalg.norm(b-btilde, ord=2)

print ('delta x = ', deltax) # 1.118033988749895
print ('delta b = ', deltab) # 0.0015000000000000568