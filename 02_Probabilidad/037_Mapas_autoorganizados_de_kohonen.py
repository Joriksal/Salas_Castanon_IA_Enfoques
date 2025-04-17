import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom  # Librería para SOM

# 1. Datos de ejemplo: 1000 colores RGB aleatorios
np.random.seed(42)  # Fijar la semilla para reproducibilidad
data = np.random.rand(1000, 3)  # Generar 1000 colores aleatorios en formato [R, G, B]

# 2. Configurar SOM: cuadrícula 10x10, dimensión de entrada=3 (RGB)
# MiniSom crea un mapa autoorganizado con una cuadrícula de 10x10 neuronas
# Cada neurona tiene un vector de pesos de dimensión 3 (correspondiente a RGB)
som = MiniSom(10, 10, 3, sigma=1.0, learning_rate=0.5, neighborhood_function='gaussian')

# 3. Inicialización aleatoria de pesos
# Inicializa los pesos de las neuronas con valores aleatorios basados en los datos
som.random_weights_init(data)

# 4. Entrenamiento (500 iteraciones)
# Entrena el SOM utilizando los datos de entrada durante 500 iteraciones
som.train_random(data, 500)

# 5. Visualización
# Crear una figura para visualizar el mapa de distancias (U-Matrix)
plt.figure(figsize=(10, 10))
plt.pcolor(som.distance_map().T, cmap='bone_r')  # Mapa de distancias (U-Matrix)
plt.colorbar()  # Barra de colores para interpretar las distancias

# Marcar los datos en el SOM
# Para cada color en los datos, encontrar la neurona ganadora y marcarla en el mapa
for i, x in enumerate(data):
    w = som.winner(x)  # Neurona ganadora para el color x
    plt.plot(w[0] + 0.5, w[1] + 0.5, 'o',  # Dibujar un círculo en la posición de la neurona ganadora
             markerfacecolor=x,  # Usar el color RGB correspondiente como relleno
             markersize=10,  # Tamaño del marcador
             markeredgewidth=0.5,  # Grosor del borde
             markeredgecolor='k')  # Color del borde (negro)

# Título del gráfico
plt.title('Mapa Autoorganizado de Kohonen: Agrupación de Colores RGB')

# Mostrar el gráfico
plt.show()