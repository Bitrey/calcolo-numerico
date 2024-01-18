"""
Esercitazione: 1
Esercizio: 2.2
Autore: Alessandro Amella
Matricola: 0001070569
"""

import math
import matplotlib.pyplot as plt
import numpy as np

def fib_iter(n):
    a = 0 # F(0)
    b = 1 # F(1)
    for i in range(n):
        c = a + b
        a = b
        b = c
    return a

print(fib_iter(10)) # 55

def fib_ratio(k):
    a = fib_iter(k)
    b = fib_iter(k - 1)
    if b == 0:
        return math.inf
    return a / b

print(fib_ratio(10)) # 1.6176470588235294

# Verify that, for a large k, {rk}k converges to the value φ = (1 + √5)/2
phi = (1 + math.sqrt(5)) / 2
def fib_ratio_convergence(k):
    return phi - fib_ratio(k)

print(fib_ratio_convergence(10)) # 0.00038692992636546464

errors = []
for n in range(30):
    r = fib_ratio(n)
    e = abs(phi - r)
    errors.append(e)
plt.plot(errors)
plt.title("Errore approssimazione di Fibonacci")
plt.xlabel("n")
plt.ylabel("Errore")
plt.show()
