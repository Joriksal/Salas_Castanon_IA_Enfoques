import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification, make_circles

# Configuración inicial
np.random.seed(42)  # Fijar la semilla para reproducibilidad
plt.style.use('seaborn-v0_8')  # Estilo de gráficos

# 1. Generación de datos linealmente separables
X_linear, y_linear = make_classification(
    n_samples=100,  # Número de muestras
    n_features=2,   # Número de características
    n_redundant=0,  # Sin características redundantes
    n_classes=2,    # Dos clases
    class_sep=2.0,  # Separación entre clases (alta para facilitar separación)
    random_state=42 # Reproducibilidad
)

# 2. Generación de datos no linealmente separables (círculos)
X_nonlinear, y_nonlinear = make_circles(
    n_samples=100,  # Número de muestras
    noise=0.05,     # Ruido en los datos
    factor=0.5,     # Relación entre radios de los círculos
    random_state=42 # Reproducibilidad
)

# Perceptrón mejorado para clasificación binaria
def train_perceptron(X, y, lr=0.1, epochs=100):
    """
    Entrena un perceptrón para separar datos linealmente separables.
    Parámetros:
        X: Matriz de características (n_samples, n_features)
        y: Etiquetas (-1 o 1)
        lr: Tasa de aprendizaje
        epochs: Número de iteraciones
    Retorna:
        weights: Pesos ajustados
        bias: Sesgo ajustado
    """
    weights = np.random.randn(X.shape[1]) * 0.01  # Pesos iniciales pequeños
    bias = 0  # Sesgo inicial
    
    for _ in range(epochs):
        z = X @ weights + bias  # Producto punto + sesgo
        y_pred = np.where(z >= 0, 1, -1)  # Predicción binaria
        errors = y - y_pred  # Errores de predicción
        weights += lr * X.T @ errors / len(X)  # Actualización de pesos
        bias += lr * np.mean(errors)  # Actualización del sesgo
    
    return weights, bias

# Escalado de datos para normalizar características
X_linear_scaled = (X_linear - X_linear.mean(axis=0)) / X_linear.std(axis=0)
w_linear, b_linear = train_perceptron(X_linear_scaled, y_linear * 2 - 1)  # Etiquetas ajustadas a -1 y 1

X_nonlinear_scaled = (X_nonlinear - X_nonlinear.mean(axis=0)) / X_nonlinear.std(axis=0)
w_nonlinear, b_nonlinear = train_perceptron(X_nonlinear_scaled, y_nonlinear * 2 - 1)

# Función para graficar la frontera de decisión
def plot_decision_boundary(X, y, weights, bias, ax, title):
    """
    Grafica la frontera de decisión del perceptrón.
    Parámetros:
        X: Matriz de características
        y: Etiquetas
        weights: Pesos del modelo
        bias: Sesgo del modelo
        ax: Ejes de matplotlib
        title: Título del gráfico
    """
    # Definir límites del gráfico
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    
    # Crear una malla de puntos
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),
        np.linspace(y_min, y_max, 100)
    )
    
    # Calcular valores de la frontera de decisión
    Z = xx * weights[0] + yy * weights[1] + bias
    Z = np.where(Z >= 0, 1, -1)
    
    # Graficar la frontera de decisión
    cmap = plt.cm.RdBu  # Paleta de colores
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-2, 0, 2])
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, 
                          edgecolors='k', s=50)  # Puntos de datos
    
    # Configuración del gráfico
    ax.set_title(title, pad=15, fontsize=12)
    ax.set_xlabel('Feature 1', labelpad=10)
    ax.set_ylabel('Feature 2', labelpad=10)
    ax.grid(True, alpha=0.3)

# Visualización de los resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Gráfico para datos linealmente separables
plot_decision_boundary(
    X_linear_scaled, y_linear, w_linear, b_linear, 
    ax1, "Linealmente Separable\n(Perceptrón converge)"
)

# Gráfico para datos no linealmente separables
plot_decision_boundary(
    X_nonlinear_scaled, y_nonlinear, w_nonlinear, b_nonlinear, 
    ax2, "No Linealmente Separable\n(Perceptrón falla)"
)

plt.tight_layout(pad=3.0)
plt.show()