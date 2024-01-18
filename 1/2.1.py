"""
Esercitazione: 1
Esercizio: 2.1
Autore: Alessandro Amella
Matricola: 0001070569
"""

import matplotlib.pyplot as plt
import numpy as np

# Array [0, 10] con 100 elementi
x = np.linspace(0, 10, 100)

# Calcolo cos(x) e sin(x)
y1 = np.cos(x)
y2 = np.sin(x)

# Mostra i grafici
plt.plot(x, y1, color='red', label='cos')
plt.plot(x, y2, color='blue', label='sin')

# Aggiungi legenda, titolo e label
plt.legend()
plt.title('Grafico seno e coseno')
plt.xlabel('x')
plt.ylabel('y')
plt.show()
