import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
from skimage import data, metrics
from scipy import signal
from numpy import fft
from utils import psf_fft, A, AT, gaussian_kernel
from scipy.optimize import minimize
from skimage.io import imread
import os
img = 'modena.png'
cur_dir = os.path.dirname(os.path.abspath(__file__))
X = imread(os.path.join(cur_dir, img), as_gray=True)
m, n = X.shape

# Genera il filtro di blur
k = gaussian_kernel(5, 1.3)
# Blur with FFT
K = psf_fft(k, 5, X.shape)
X_blurred = A(X, K)

# rumore gaussiano con deviazione standard nell’ intervallo (0, 0, 05].
sigma = 0.05
noise = np.random.normal(0, sigma, X.shape)

# Aggiungi blur e rumore
y = X_blurred + noise
PSNR = metrics.peak_signal_noise_ratio(X, y)
ATy = AT(y, K)

# Downsampling con fattore di sottocampionamento S
S=4
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
# lambdas = [0.05, 0.06, 0.07, 0.08]
lambdas = np.linspace(0.05, 0.08, 8)
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
plt.subplot(2, 5, 1)
plt.imshow(X, cmap='gray')
plt.title('Originale')
plt.subplot(2, 5, 2)
y_d = A(X_d, K_d)
plt.imshow(y_d, cmap='gray')
plt.title('Corrotta')
for i, X_curr in enumerate(images):
    plt.subplot(2, 5, i + 3)
    plt.imshow(X_curr, cmap='gray')
    plt.title(f'$\lambda = {lambdas[i]:.3f}$')
plt.show()
