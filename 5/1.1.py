"""
Esercitazione: 5
Esercizio: 1.1, 1.2
Autore: Alessandro Amella
Matricola: 0001070569
"""

import numpy as np
import matplotlib.pyplot as plt

# Exercise 1
# Function approssimazioni successive
def succ_app(f, g, tolf, tolx, maxit, xTrue, x0=0):
  i=0
  err=np.zeros(maxit+1, dtype=np.float64)
  err[0]=tolx+1
  vecErrore=np.zeros(maxit+1, dtype=np.float64)
  vecErrore[0] = np.abs(x0-xTrue)
  x=x0

  while (err[i]>tolx and i<maxit): # scarto assoluto tra iterati
    x_new=g(x)
    err[i+1]=np.abs(x_new-x)
    vecErrore[i+1]=np.abs(x_new-xTrue)
    i=i+1
    x=x_new
  err=err[0:i]      
  vecErrore = vecErrore[0:i]
  return (x, i, err, vecErrore) 

def newton(f, df, tolf, tolx, maxit, xTrue, x0=0):
    g = lambda x: x-f(x)/df(x)
    (x, i, err, vecErrore) = succ_app(f, g, tolf, tolx, maxit, xTrue, x0)
    return (x, i, err, vecErrore)

f = lambda x: np.exp(x)-x**2
df = lambda x: np.exp(x)-2*x
g1 = lambda x: x-f(x)*np.exp(x/2)
g2 = lambda x: x-f(x)*np.exp(-x/2)

# first, plot and show f

xTrue = -0.703467
fTrue = f(xTrue)
print('fTrue = ', fTrue) # 8.035078391532835e-07

xplot = np.linspace(-1, 1)
fplot = f(xplot)

plt.plot(xplot,fplot, label='f(x)')
plt.plot(xTrue,fTrue, 'or', label='$x^*$')

plt.legend()
plt.grid()
plt.show()

plt.plot(xplot,fplot)
# plt.plot(xTrue,fTrue, 'or', label='True')

tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 0

[sol_g1, iter_g1, err_g1, vecErrore_g1]=succ_app(f, g1, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g1 \n x =',sol_g1,'\n iter_new=', iter_g1)

plt.plot(sol_g1,f(sol_g1), '^', label='g1')

[sol_g2, iter_g2, err_g2, vecErrore_g2]=succ_app(f, g2, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g2 \n x =',sol_g2,'\n iter_new=', iter_g2)

plt.plot(sol_g2,f(sol_g2), 'o', label='g2')

[sol_newton, iter_newton, err_newton, vecErrore_newton]=newton(f, df, tolf, tolx, maxit, xTrue, x0)
print('Metodo Newton \n x =',sol_newton,'\n iter_new=', iter_newton)

plt.plot(sol_newton,f(sol_newton), '+', label='Newton')
plt.legend()
plt.grid()
plt.show()

# GRAFICO Errore vs Iterazioni

# g1
plt.plot(vecErrore_g1, '.-', color='blue')
# g2
plt.plot(vecErrore_g2[:20], '.-', color='green')
# Newton
plt.plot(vecErrore_newton, '.-', color='red')

plt.legend( ("g1", "g2", "newton"))
plt.xlabel('iter')
plt.ylabel('errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()

# 1.2
f = lambda x: x**3+4*x*np.cos(x)-2
df = lambda x: 3*x**2+4*np.cos(x)-4*x*np.sin(x)
g = lambda x: (2-x**3)/(4*np.cos(x))

xplot = np.linspace(0, 2)
fplot = f(xplot)

xTrue = 0.5369
fTrue = f(xTrue)
print('fTrue = ', fTrue)

plt.plot(xplot,fplot)
plt.plot(xTrue,fTrue, '^r', label='True')

tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 1

[sol_g, iter_g, err_g, vecErrore_g]=succ_app(f, g, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g \n x =',sol_g,'\n iter_new=', iter_g)

plt.plot(sol_g,f(sol_g), 'o', label='g=$\\frac{2-x^3}{4cos(x)}$')

[sol_newton, iter_newton, err_newton, vecErrore_newton]=newton(f, df, tolf, tolx, maxit, xTrue, x0)
print('Metodo Newton \n x =',sol_newton,'\n iter_new=', iter_newton)

plt.plot(sol_newton,f(sol_newton), '+b', label='Newton')
plt.grid()
plt.legend()
plt.show()

# GRAFICO Errore vs Iterazioni

# g
plt.plot(vecErrore_g, '.-', color='blue')
# Newton
plt.plot(vecErrore_newton, '.-', color='red')

plt.legend( ("g", "newton"))
plt.xlabel('iter')
plt.ylabel('errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()

# f = x - x**(1/3) - 2

f = lambda x: x-x**(1/3)-2
df = lambda x: 1-1/(3*x**(2/3))
g = lambda x: x**(1/3)+2

xTrue = 3.5213
fTrue = f(xTrue)
print('fTrue = ', fTrue)

xplot = np.linspace(3, 5)
fplot = f(xplot)

plt.plot(xplot,fplot)
plt.plot(xTrue,fTrue, '^r', label='True')

tolx= 10**(-10)
tolf = 10**(-6)
maxit=100
x0= 4

[sol_g, iter_g, err_g, vecErrore_g]=succ_app(f, g, tolf, tolx, maxit, xTrue, x0)
print('Metodo approssimazioni successive g \n x =',sol_g,'\n iter_new=', iter_g)

plt.plot(sol_g,f(sol_g), 'o', label='$g=x^{1/3}+2$')

[sol_newton, iter_newton, err_newton, vecErrore_newton]=newton(f, df, tolf, tolx, maxit, xTrue, x0)
print('Metodo Newton \n x =',sol_newton,'\n iter_new=', iter_newton)

plt.plot(sol_newton,f(sol_newton), '+b', label='Newton')
plt.grid()
plt.legend()
plt.show()

# GRAFICO Errore vs Iterazioni

# g
plt.plot(vecErrore_g, '.-', color='blue')
# Newton
plt.plot(vecErrore_newton, '.-', color='red')

plt.legend( ("g=$x^{1/3}+2$", "newton"))
plt.xlabel('iter')
plt.ylabel('errore')
plt.title('Errore vs Iterazioni')
plt.grid()
plt.show()
