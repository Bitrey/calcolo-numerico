"""
Esercitazione: 1
Esercizio: 1.1
Autore: Alessandro Amella
Matricola: 0001070569
"""

import math

def machine_epsilon(func=float):
    eps = func(1)
    while func(1) + func(eps) != func(1):
        eps_last = eps
        eps = func(eps) / func(2)
    return eps_last

print(machine_epsilon(float)) # 2.220446049250313e-16

def mantissa_digits(eps):
    return -math.log10(eps)

print(mantissa_digits(machine_epsilon(float))) # 15.653559774527022

import numpy as np
print(machine_epsilon(np.float16)) # 0.000977
print(machine_epsilon(np.float32)) # 1.1920929e-07

print(np.finfo(float).eps) # 2.220446049250313e-16
print(np.finfo(np.float16).eps) # 0.000977
print(np.finfo(np.float32).eps) # 1.1920929e-07
