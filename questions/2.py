import numpy as np
from scipy.linalg import cholesky

n = 5
A = np.diag(9 * np.ones(n)) + np.diag(-4 * np.ones(n-1), 1) + np.diag(-4 * np.ones(n-1), -1)
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