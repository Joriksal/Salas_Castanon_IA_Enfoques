# --- Importación de librerías ---
import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos/matrices.
import matplotlib.pyplot as plt  # Librería para generar gráficos y visualizaciones.
from sklearn.datasets import make_moons  # Función para generar datos de ejemplo con forma de "lunas".

# --- Configuración inicial ---
# Se establece una semilla para garantizar la reproducibilidad de los resultados.
np.random.seed(42)

# Generación de datos de ejemplo:
# `make_moons` crea un conjunto de datos con dos clases en forma de lunas entrelazadas.
# `n_samples=200` indica que se generarán 200 puntos.
# `noise=0.2` agrega ruido aleatorio para que los datos no sean perfectamente lineales.
# `random_state=42` asegura que los datos sean reproducibles.
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)

# Reformatear las etiquetas `y` para que sean columnas (necesario para cálculos posteriores).
y = y.reshape(-1, 1)  # `-1` ajusta automáticamente el número de filas.

# --- Definición de la arquitectura de la red neuronal ---
class MLP:
    """
    Clase que implementa una red neuronal multicapa (MLP) simple con:
    - Una capa oculta con 5 neuronas y función de activación ReLU.
    - Una capa de salida con 1 neurona y función de activación sigmoide.
    """

    def __init__(self):
        """
        Constructor de la clase. Inicializa los pesos y sesgos de la red neuronal.
        """
        # Pesos y sesgos de la capa oculta (2 entradas → 5 neuronas).
        self.W1 = np.random.randn(2, 5) * 0.01  # Pesos inicializados aleatoriamente con distribución normal.
        self.b1 = np.zeros((1, 5))  # Sesgos inicializados en cero.

        # Pesos y sesgos de la capa de salida (5 neuronas → 1 salida).
        self.W2 = np.random.randn(5, 1) * 0.01  # Pesos inicializados aleatoriamente.
        self.b2 = np.zeros((1, 1))  # Sesgos inicializados en cero.

    def relu(self, x):
        """
        Función de activación ReLU (Rectified Linear Unit).
        Devuelve el valor máximo entre 0 y el valor de entrada.
        """
        return np.maximum(0, x)

    def sigmoid(self, x):
        """
        Función de activación sigmoide.
        Convierte cualquier valor en un rango entre 0 y 1, útil para problemas de clasificación binaria.
        """
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        """
        Propagación hacia adelante: calcula las salidas de la red neuronal.
        """
        # Cálculo de la salida de la capa oculta.
        self.z1 = X @ self.W1 + self.b1  # Producto punto entre entradas y pesos, más el sesgo.
        self.a1 = self.relu(self.z1)  # Aplicación de la función de activación ReLU.

        # Cálculo de la salida de la capa final.
        self.z2 = self.a1 @ self.W2 + self.b2  # Producto punto entre salidas de la capa oculta y pesos de la capa final.
        self.a2 = self.sigmoid(self.z2)  # Aplicación de la función de activación sigmoide.
        return self.a2  # Devuelve la probabilidad de pertenecer a la clase positiva.

    def backward(self, X, y, lr):
        """
        Propagación hacia atrás: calcula los gradientes y actualiza los pesos y sesgos.
        """
        m = len(X)  # Número de muestras en el conjunto de datos.

        # Gradiente de la capa de salida.
        dz2 = self.a2 - y  # Diferencia entre la predicción y la etiqueta real.
        dW2 = (self.a1.T @ dz2) / m  # Gradiente de los pesos de la capa de salida.
        db2 = np.sum(dz2, axis=0, keepdims=True) / m  # Gradiente de los sesgos de la capa de salida.

        # Gradiente de la capa oculta.
        da1 = dz2 @ self.W2.T  # Propagación del error hacia la capa oculta.
        dz1 = da1 * (self.z1 > 0)  # Derivada de la función ReLU.
        dW1 = (X.T @ dz1) / m  # Gradiente de los pesos de la capa oculta.
        db1 = np.sum(dz1, axis=0, keepdims=True) / m  # Gradiente de los sesgos de la capa oculta.

        # Actualización de los pesos y sesgos usando gradiente descendente.
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# --- Entrenamiento de la red neuronal ---
mlp = MLP()  # Crear una instancia de la red neuronal.
lr = 0.1  # Tasa de aprendizaje (qué tan grandes son los pasos en la actualización de parámetros).
epochs = 3000  # Número de iteraciones (épocas) para entrenar la red.
loss_history = []  # Lista para almacenar el historial de pérdida.

for epoch in range(epochs):
    # Propagación hacia adelante y cálculo de la pérdida.
    y_pred = mlp.forward(X)  # Predicción de la red neuronal.
    # Pérdida: entropía cruzada binaria (métrica para problemas de clasificación).
    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    loss_history.append(loss)  # Almacenar la pérdida actual.

    # Propagación hacia atrás y actualización de parámetros.
    mlp.backward(X, y, lr)

    # Mostrar la pérdida cada 500 épocas.
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Visualización de resultados ---
plt.figure(figsize=(15, 5))  # Crear una figura con tamaño personalizado.

# 1. Gráfico de pérdida durante el entrenamiento.
plt.subplot(1, 2, 1)  # Primer gráfico (1 fila, 2 columnas, posición 1).
plt.plot(loss_history)  # Graficar el historial de pérdida.
plt.title("Pérdida durante el entrenamiento")  # Título del gráfico.
plt.xlabel("Época")  # Etiqueta del eje X.
plt.ylabel("Entropía Cruzada")  # Etiqueta del eje Y.

# 2. Frontera de decisión de la red neuronal.
plt.subplot(1, 2, 2)  # Segundo gráfico (1 fila, 2 columnas, posición 2).
# Definir los límites del gráfico.
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# Crear una malla de puntos para evaluar la red.
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)  # Predicciones para cada punto de la malla.

# Dibujar la frontera de decisión.
plt.contourf(xx, yy, Z > 0.5, alpha=0.3, cmap='coolwarm')  # Frontera de decisión (colores).
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm', edgecolors='k')  # Puntos de datos.
plt.title("Frontera de Decisión MLP")  # Título del gráfico.

plt.tight_layout()  # Ajustar el diseño para evitar superposición.
plt.show()  # Mostrar los gráficos.