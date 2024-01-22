import numpy as np
import matplotlib.pyplot as plt
from skimage import data

A = data.camera()
U, s, Vh = np.linalg.svd(A)
p_max = 20
plt.figure(figsize=(p_max, p_max // 2))
for i in range(1, p_max + 1, 2):
    A_p = np.dot(U[:, :i], np.dot(np.diag(s[:i]), Vh[:i, :]))

    plt.subplot(2, p_max // 4, (i + 1) // 2)
    plt.imshow(A_p, cmap='gray')
    plt.title('p = {}'.format(i))

plt.show()
