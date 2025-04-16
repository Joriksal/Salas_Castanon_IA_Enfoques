import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# --- Configuración ---
# Se establece una semilla para reproducibilidad
np.random.seed(42)

# Se generan datos de ejemplo con forma de "lunas" para clasificación binaria
X, y = make_moons(n_samples=200, noise=0.2, random_state=42)
y = y.reshape(-1, 1)  # Reformatear etiquetas para que sean columnas

# --- Arquitectura MLP ---
class MLP:
    def __init__(self):
        # Inicialización de pesos y sesgos para la red
        # Capa oculta: 2 entradas → 5 neuronas
        self.W1 = np.random.randn(2, 5) * 0.01
        self.b1 = np.zeros((1, 5))
        # Capa de salida: 5 neuronas → 1 salida
        self.W2 = np.random.randn(5, 1) * 0.01
        self.b2 = np.zeros((1, 1))

    def relu(self, x):
        # Función de activación ReLU
        return np.maximum(0, x)

    def sigmoid(self, x):
        # Función de activación Sigmoide
        return 1 / (1 + np.exp(-x))

    def forward(self, X):
        # Propagación hacia adelante
        # Cálculo de la salida de la capa oculta
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        # Cálculo de la salida de la capa final
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = self.sigmoid(self.z2)  # Probabilidad de clase positiva
        return self.a2

    def backward(self, X, y, lr):
        # Propagación hacia atrás (backpropagation)
        m = len(X)  # Número de muestras

        # Gradiente de la capa de salida
        dz2 = self.a2 - y
        dW2 = (self.a1.T @ dz2) / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        # Gradiente de la capa oculta
        da1 = dz2 @ self.W2.T
        dz1 = da1 * (self.z1 > 0)  # Derivada de ReLU
        dW1 = (X.T @ dz1) / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        # Actualización de pesos y sesgos
        self.W2 -= lr * dW2
        self.b2 -= lr * db2
        self.W1 -= lr * dW1
        self.b1 -= lr * db1

# --- Entrenamiento ---
mlp = MLP()  # Instancia de la red neuronal
lr = 0.1  # Tasa de aprendizaje
epochs = 3000  # Número de épocas
loss_history = []  # Historial de pérdida

for epoch in range(epochs):
    # Propagación hacia adelante y cálculo de la pérdida
    y_pred = mlp.forward(X)
    # Pérdida: Entropía cruzada binaria
    loss = -np.mean(y * np.log(y_pred + 1e-9) + (1 - y) * np.log(1 - y_pred + 1e-9))
    loss_history.append(loss)

    # Propagación hacia atrás y actualización de parámetros
    mlp.backward(X, y, lr)

    # Mostrar pérdida cada 500 épocas
    if epoch % 500 == 0:
        print(f"Epoch {epoch}, Loss: {loss:.4f}")

# --- Visualización ---
plt.figure(figsize=(15, 5))

# 1. Gráfico de pérdida
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Entropía Cruzada")

# 2. Frontera de decisión
plt.subplot(1, 2, 2)
# Definir límites del gráfico
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
# Crear una malla de puntos para evaluar la red
xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
Z = mlp.forward(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)

# Dibujar la frontera de decisión
plt.contourf(xx, yy, Z > 0.5, alpha=0.3, cmap='coolwarm')
plt.scatter(X[:, 0], X[:, 1], c=y.flatten(), cmap='coolwarm', edgecolors='k')
plt.title("Frontera de Decisión MLP")

plt.tight_layout()
plt.show()