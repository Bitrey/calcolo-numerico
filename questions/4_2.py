import numpy as np
import matplotlib.pyplot as plt
from skimage import data
from skimage.io import imread
import os

A = data.camera()
U, s, Vh = np.linalg.svd(A)
p_max = 100
A_p = np.zeros(A.shape)
err_rel = np.zeros((p_max))
c = np.zeros((p_max))
for i in range(p_max):
    ui = U[:, :i+1]
    vi = Vh[:i+1, :]
    
    A_p = np.dot(ui, np.dot(np.diag(s[:i+1]), vi))

    err_rel[i] = np.linalg.norm(A - A_p) / np.linalg.norm(A)
    c[i] = 1 / (i + 1) * (min(A.shape) - 1)

plt.figure(figsize=(10, 5))

fig1 = plt.subplot(1, 2, 1)
fig1.plot(err_rel, 'o-')
plt.title('Errore relativo')

fig2 = plt.subplot(1, 2, 2)
fig2.plot(c, 'o-')
plt.title('Fattore di compressione')

plt.show()
