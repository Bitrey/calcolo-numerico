"""
Esercitazione: 2
Esercizio: 2.1, 2.2
Autore: Alessandro Amella
Matricola: 0001070569
"""

"""2.2 Choleski con matrice di Hilbert"""
import matplotlib.pyplot as plt
import numpy as np
import scipy
# help (scipy)
import scipy.linalg
# help (scipy.linalg)
# help (scipy.linalg.cholesky)
# help (scipy.linalg.hilbert)

# crazione dati e problema test
n = 5
A = scipy.linalg.hilbert(n)
x = np.ones((n,1))
b = A @ x

condA = np.linalg.cond(A, 2)

print('x: \n', x , '\n')
print('x.shape: ', x.shape, '\n' )
print('b: \n', b , '\n')
print('b.shape: ', b.shape, '\n' )
print('A: \n', A, '\n')
print('A.shape: ', A.shape, '\n' )
print('K(A)=', condA, '\n')

# decomposizione di Choleski
L = scipy.linalg.cholesky(A, lower=True)
print('L:', L, '\n')

print('L.T*L =', scipy.linalg.norm(A-np.matmul(np.transpose(L),L)))
print('err = ', scipy.linalg.norm(A-np.matmul(np.transpose(L),L), 'fro'))

y = scipy.linalg.solve_triangular(L, b, lower=True)
my_x = scipy.linalg.solve_triangular(L.T, y, lower=False)
print('my_x = \n ', my_x)

print('norm =', np.linalg.norm(x-my_x, 'fro'))


K_A = np.zeros((6,1))
Err = np.zeros((6,1))

for n in np.arange(5,11):
    # crazione dati e problema test
    A = scipy.linalg.hilbert(n)
    x = np.ones((n,1))
    b = A @ x
    
    # numero di condizione 
    K_A[n-5] = np.linalg.cond(A, 2)
    
    # fattorizzazione 
    L = scipy.linalg.cholesky(A, lower=True)
    y = scipy.linalg.solve_triangular(L, b, lower=True)
    my_x = scipy.linalg.solve_triangular(L.T, y, lower=False)
    
    # errore relativo
    Err[n-5] = np.linalg.norm(x-my_x, 'fro')/np.linalg.norm(x, 'fro')
  
xplot = np.arange(5,11)

# grafico del numero di condizione vs dim
plt.semilogy(xplot, K_A)
plt.title('CONDIZIONAMENTO DI A ')
plt.xlabel('dimensione matrice: n')
plt.ylabel('K(A)')
plt.show()


# grafico errore in norma 2 in funzione della dimensione del sistema
plt.plot(xplot, Err)
plt.title('Errore relativo')
plt.xlabel('dimensione matrice: n')
plt.ylabel('Err= ||my_x-x||/||x||')
plt.show()