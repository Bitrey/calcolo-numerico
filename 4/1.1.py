"""
Esercitazione: 4
Esercizio: 1.1
Autore: Alessandro Amella
Matricola: 0001070569
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import lu_factor as LUdec 

# Exercise 1

m = 100
n = 10

A = np.random.rand(m, n)

alpha_test = np.full(n, 0.5) # esempio con α = 0.5
y = A @ alpha_test # y = A*α

print('alpha test', alpha_test) # [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]

ATA = A.T@A
ATy = A.T@y

lu, piv = LUdec(ATA)
alpha_LU = scipy.linalg.lu_solve((lu,piv), ATy)

print('alpha LU', alpha_LU) # [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]

L = scipy.linalg.cholesky(ATA)
x = scipy.linalg.solve_triangular(np.transpose(L), ATy, lower=True)
alpha_chol = scipy.linalg.solve_triangular(L, x, lower=False)

print('alpha chol', alpha_chol) # [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]

U, s, Vh = scipy.linalg.svd(A)

print('Shape of U:', U.shape) # (100, 100)
print('Shape of s:', s.shape) # (10,)
print('Shape of V:', Vh.T.shape) # (10, 10)

alpha_svd = np.zeros(s.shape)

for i in range(n):
  ui = U[:, i]
  vi = Vh[i, :]

  alpha_svd = alpha_svd + (ui.T@y/s[i])*vi

print('alpha SVD', alpha_svd) # [0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5 0.5]

# Calcolare l'errore relativo delle soluzioni trovate, rispetto al vettore \(\boldsymbol{\alpha}\), soluzione esatta utilizzata per generare il problema test.
err_rel_LU = np.linalg.norm(alpha_LU - alpha_test) / np.linalg.norm(alpha_test)
err_rel_chol = np.linalg.norm(alpha_chol - alpha_test) / np.linalg.norm(alpha_test)
err_rel_svd = np.linalg.norm(alpha_svd - alpha_test) / np.linalg.norm(alpha_test)

print('\n')
print('Errore relativo LU', err_rel_LU) # 6.7143255907214105e-15
print('Errore relativo chol', err_rel_chol) # 7.27852596108914e-15
print('Errore relativo SVD', err_rel_svd) # 1.5570853182758493e-15