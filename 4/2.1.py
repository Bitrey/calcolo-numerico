"""
Esercitazione: 4
Esercizio: 2.1
Autore: Alessandro Amella
Matricola: 0001070569
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from scipy.linalg import lu_factor as LUdec 

# Exercise 2
case = 2
m = 10
m_plot = 100

# Grado polinomio approssimante
n = 5

if case==0:
    x = np.linspace(-1,1,m)
    y = np.exp(x/2)
elif case==1:
    x = np.linspace(-1,1,m)
    y = 1/(1+25*(x**2))
elif case==2:
    x = np.linspace(0,2*np.pi,m)
    y = np.sin(x)+np.cos(x)


A = np.zeros((m, n+1))

for i in range(n+1):
  A[:, i] = x**i
  
U, s, Vh = scipy.linalg.svd(A)


alpha_svd = np.zeros(n+1)

for i in range(n+1):
  ui = U[:, i]
  vi = Vh[i, :]

  alpha_svd = alpha_svd + (ui.T@y/s[i])*vi

print(alpha_svd)


x_plot = np.linspace(x[0], x[-1], m_plot)
A_plot = np.zeros((m_plot, n+1))

for i in range(n+1):
  A_plot[:, i] = x_plot**i

y_interpolation = A_plot@alpha_svd

plt.plot(x, y, 'o')
plt.plot(x_plot, y_interpolation, 'r')
plt.xlabel('x')
plt.ylabel('y')
plt.title(f'Interpolazione polinomiale di grado {n}')
plt.grid()
plt.show()


res = np.linalg.norm(y - A@alpha_svd)
print('Residual: ', res)