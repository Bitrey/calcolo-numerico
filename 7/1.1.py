'''
Esercitazione: 7
Esercizio: 1.1
Autore: Alessandro Amella
Matricola: 0001070569
'''

import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy import signal
from numpy import fft
from utils import psf_fft, A, AT, gaussian_kernel

# Immagine in floating point con valori tra 0 e 1
X = data.camera() / 255
m, n = X.shape

# Genera il filtro di blur
k = gaussian_kernel(24, 3)
# Blur with FFT
K = psf_fft(k, 24, X.shape)
X_blurred = A(X, K)

# Genera il rumore
sigma = 0.02
np.random.seed(42)
noise = np.random.normal(size=X.shape) * sigma

# Aggiungi blur e rumore
y = X_blurred + noise
PSNR = metrics.peak_signal_noise_ratio(X, y)
ATy = AT(y, K)

from scipy.optimize import minimize

# Regolarizzazione: x = argmin 0.5 * ||Ax - y||^2 + 0.5 * L * ||x||^2
# Funzione da minimizzare
def f(x, L):
    nsq = np.sum(np.square(x))
    x  = x.reshape((m, n))
    Ax = A(x, K)
    return 0.5 * np.sum(np.square(Ax - y)) + 0.5 * L * nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m, n)
    ATAx = AT(A(x,K),K)
    d = ATAx - ATy
    return d.reshape(m * n) + Lx

x0 = y.reshape(m*n)
# lambdas = [0.02,0.03,0.04, 0.06]
lambdas = [] # DEBUG
PSNRs = []
images = []

# # inanzitutto mostriamo blur vs lambda=0.02 con subplot
# L=0.02
# max_iter = 50
# res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})
# X_curr = res.x.reshape(X.shape)
# PSNR = metrics.peak_signal_noise_ratio(X, X_curr)
# print(f'PSNR for lambda={L}: {PSNR}')
# plt.figure()
# plt.subplot(1,2,1)
# plt.imshow(X_blurred, cmap='gray')
# plt.title('Corrotta')
# plt.subplot(1,2,2)
# plt.imshow(X_curr, cmap='gray')
# plt.title(f'Ricostruita con $\lambda = {L:.2f}$ (PSNR: {PSNR:.4f}))')
# plt.show()


# plt.figure(figsize=(30, 10))

# plt.subplot(1, len(lambdas) + 2, 1).imshow(X, cmap='gray', vmin=0, vmax=1)
# plt.title("Originale")
# plt.xticks([]), plt.yticks([])
# plt.subplot(1, len(lambdas) + 2, 2).imshow(y, cmap='gray', vmin=0, vmax=1)
# plt.title("Corrotta")
# plt.xticks([]), plt.yticks([])


# Ricostruzione per diversi valori del parametro di regolarizzazione
for i, L in enumerate(lambdas):
    # Esegui la minimizzazione con al massimo 50 iterazioni
    max_iter = 50
    res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})

    # Aggiungi la ricostruzione nella lista images
    X_curr = res.x.reshape(X.shape)
    images.append(X_curr)

    # Stampa il PSNR per il valore di lambda attuale
    PSNR = metrics.peak_signal_noise_ratio(X, X_curr)
    PSNRs.append(PSNR)
    print(f'PSNR for lambda={L}: {PSNR}')

# # Stampa il PSNR per il valore di lambda attuale
# plt.figure()
# plt.plot(lambdas, PSNRs)
# plt.xlabel('Lambda')
# plt.ylabel('PSNR')
# plt.title('PSNR vs lambda')
# plt.show()

# # Stampa le ricostruzioni
# plt.figure()
# for i, X_curr in enumerate(images):
#     plt.subplot(2, 2, i + 1)
#     plt.imshow(X_curr, cmap='gray')
#     plt.title(f'lambda={lambdas[i]}')
# plt.show()

# ora usiamo downsampling con S = 2,4,8,16
# Regolarizzazione: x = argmin 0.5 * ||SAx - y||^2 + 0.5 * L * ||x||^2

# Downsampling con S=2
S=8
X_d = X[::S,::S]
y_d = y[::S,::S]
K_d = K[::S,::S]
ATy_d = AT(y_d, K_d)

# Funzione da minimizzare
def f(x, L):
    nsq = np.sum(np.square(x))
    x  = x.reshape((m//S, n//S))
    Ax = A(x, K_d)
    return 0.5 * np.sum(np.square(Ax - y_d)) + 0.5 * L * nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m//S, n//S)
    ATAx = AT(A(x,K_d),K_d)
    d = ATAx - ATy_d
    return d.reshape(m//S * n//S) + Lx

x0 = y_d.reshape(m//S*n//S)
# lambdas = [0.03,0.04, 0.06, 0.08]
lambdas = [0.06, 0.08, 0.1, 0.12]
PSNRs = []
images = []

# Ricostruzione per diversi valori del parametro di regolarizzazione
for i, L in enumerate(lambdas):
    # Esegui la minimizzazione con al massimo 50 iterazioni
    max_iter = 50
    res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})

    # Aggiungi la ricostruzione nella lista images
    X_curr = res.x.reshape(X_d.shape)
    images.append(X_curr)

    # Stampa il PSNR per il valore di lambda attuale
    PSNR = metrics.peak_signal_noise_ratio(X_d, X_curr)
    PSNRs.append(PSNR)
    print(f'PSNR for lambda={L}: {PSNR}')

# Stampa il PSNR per il valore di lambda attuale
plt.figure()
plt.plot(lambdas, PSNRs)
plt.xlabel('$\lambda$')
plt.ylabel('PSNR')
plt.title(f'PSNR per $\lambda$ con $S={S}$')
plt.show()

# Stampa originale e le ricostruzioni
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('Originale')
plt.subplot(2, 3, 2)
y_d = A(X_d, K_d)
plt.imshow(y_d, cmap='gray')
plt.title('Corrotta')
for i, X_curr in enumerate(images):
    plt.subplot(2, 3, i + 3)
    plt.imshow(X_curr, cmap='gray')
    plt.title(f'$\lambda = {lambdas[i]}$ (PSNR: {PSNRs[i]:.4f})')
plt.show()

from skimage.io import imread
import os
# img = 'modena.png'
img = 'sand_microscope.png'
cur_dir = os.path.dirname(os.path.abspath(__file__))
X = imread(os.path.join(cur_dir, img), as_gray=True)
m, n = X.shape

# Genera il filtro di blur
k = gaussian_kernel(5, 1.3)
# Blur with FFT
K = psf_fft(k, 5, X.shape)
X_blurred = A(X, K)

# Genera il rumore
# sigma = 0.02
# np.random.seed(42)
# noise = np.random.normal(size=X.shape) * sigma

# rumore gaussiano con deviazione standard nell’ intervallo (0, 0, 05].
sigma = 0.05
noise = np.random.normal(0, sigma, X.shape)

# Aggiungi blur e rumore
y = X_blurred + noise
PSNR = metrics.peak_signal_noise_ratio(X, y)
ATy = AT(y, K)

# Downsampling con S=2
S=8
X_d = X[::S,::S]
y_d = y[::S,::S]
K_d = K[::S,::S]
ATy_d = AT(y_d, K_d)

# Funzione da minimizzare
def f(x, L):
    nsq = np.sum(np.square(x))
    x  = x.reshape((m//S, n//S))
    Ax = A(x, K_d)
    return 0.5 * np.sum(np.square(Ax - y_d)) + 0.5 * L * nsq

# Gradiente della funzione da minimizzare
def df(x, L):
    Lx = L * x
    x = x.reshape(m//S, n//S)
    ATAx = AT(A(x,K_d),K_d)
    d = ATAx - ATy_d
    return d.reshape(m//S * n//S) + Lx

x0 = y_d.reshape(m//S*n//S)
# lambdas = [0.03,0.04, 0.05, 0.06]
# lambdas = [0.06, 0.08, 0.1, 0.12]
# lambdas = [0.12, 0.14, 0.16, 0.18]
lambdas = [0.10, 0.12, 0.14, 0.16]
PSNRs = []
MSEs = []
images = []

# Ricostruzione per diversi valori del parametro di regolarizzazione
for i, L in enumerate(lambdas):
    # Esegui la minimizzazione con al massimo 50 iterazioni
    max_iter = 50
    res = minimize(f, x0, (L), method='CG', jac=df, options={'maxiter':max_iter})

    # Aggiungi la ricostruzione nella lista images
    X_curr = res.x.reshape(X_d.shape)
    images.append(X_curr)

    # Stampa il PSNR per il valore di lambda attuale
    PSNR = metrics.peak_signal_noise_ratio(X_d, X_curr)
    PSNRs.append(PSNR)

    # Stampa il MSE per il valore di lambda attuale
    MSE = metrics.mean_squared_error(X_d, X_curr)
    MSEs.append(MSE)

    print(f'PSNR for lambda={L}: {PSNR}')

# Stampa il PSNR ed MSE per il valore di lambda attuale
ax1 = plt.subplot(1, 2, 1)
ax1.plot(lambdas, PSNRs)
plt.title('PSNR per $\lambda$')
plt.ylabel("PSNR")
plt.xlabel('$\lambda$')
ax2 = plt.subplot(1, 2, 2)
ax2.plot(lambdas, MSEs)
plt.title('MSE per $\lambda$')
plt.ylabel("MSE")
plt.xlabel('$\lambda$')
plt.show()

# Stampa originale e le ricostruzioni
plt.figure()
plt.subplot(2, 3, 1)
plt.imshow(X, cmap='gray')
plt.title('Originale')
plt.subplot(2, 3, 2)
y_d = A(X_d, K_d)
plt.imshow(y_d, cmap='gray')
plt.title('Corrotta')
for i, X_curr in enumerate(images):
    plt.subplot(2, 3, i + 3)
    plt.imshow(X_curr, cmap='gray')
    plt.title(f'$\lambda = {lambdas[i]}$ (PSNR: {PSNRs[i]:.4f})')
plt.show()


