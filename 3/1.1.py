"""
Esercitazione: 3
Esercizio: 1.1
Autore: Alessandro Amella
Matricola: 0001070569
"""

import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg
from skimage import data
from skimage.io import imread
import os

# help(data)

A = data.camera()
# A = data...
# A = imread...

# 2.2: uso immagine ./phantom.png
# img = 'phantom.png'
# cur_dir = os.path.dirname(os.path.abspath(__file__))
# A = imread(os.path.join(cur_dir, img), as_gray=True)

print(type(A))
print(A.shape)


plt.imshow(A, cmap='gray')
plt.show()


U, s, Vh = np.linalg.svd(A)

print('Shape of U:', U.shape)
print('Shape of s:', s.shape)
print('Shape of V:', Vh.T.shape)

A_p = np.zeros(A.shape)
p_max = 10


for i in range(p_max):
  ui = U[:, i]
  vi = Vh[i, :]

  A_p = A_p + s[i] * np.outer(ui, vi)

err_rel = np.linalg.norm(A - A_p) / np.linalg.norm(A)
c = 1 / p_max * (min(A.shape) - 1)

print('\n')
print('L\'errore relativo della ricostruzione di A è', err_rel) # 0.36547368241427036
print('Il fattore di compressione è c=', c) # 47.900000000000006


plt.figure(figsize=(20, 10))

fig1 = plt.subplot(1, 2, 1)
fig1.imshow(A, cmap='gray')
plt.title('True image')

fig2 = plt.subplot(1, 2, 2)
fig2.imshow(A_p, cmap='gray')
plt.title('Reconstructed image with p =' + str(p_max))

plt.show()



# al variare di p
p_max = 100
A_p = np.zeros(A.shape)
err_rel = np.zeros((p_max))
c = np.zeros((p_max))

for i in range(p_max):
  ui = U[:, i]
  vi = Vh[i, :]

  A_p = A_p + s[i] * np.outer(ui, vi)

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