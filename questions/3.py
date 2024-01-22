import numpy as np
from scipy.linalg import cholesky
from scipy.linalg import hilbert

n = 5
A = hilbert(n)

b = np.ones(n)

K_A = np.linalg.cond(A)

L = cholesky(A, lower=True)
x = np.linalg.solve(L, b)

print("A")
print(A)
print("\nK_A =", K_A)
print("\nL")
print(L)
print("\nx =", x)