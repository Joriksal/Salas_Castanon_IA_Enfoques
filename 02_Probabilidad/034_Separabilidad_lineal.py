# Importación de librerías necesarias
import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos
import matplotlib.pyplot as plt  # Librería para visualización de gráficos
from sklearn.datasets import make_classification, make_circles  # Generación de datos sintéticos

# Configuración inicial
np.random.seed(42)  # Fijar la semilla para reproducibilidad de resultados aleatorios
plt.style.use('seaborn-v0_8')  # Aplicar un estilo predefinido a los gráficos

# 1. Generación de datos linealmente separables
X_linear, y_linear = make_classification(
    n_samples=100,  # Número de muestras (filas de datos)
    n_features=2,   # Número de características (columnas de datos)
    n_redundant=0,  # Sin características redundantes (todas son útiles)
    n_classes=2,    # Número de clases (binario: 0 y 1)
    class_sep=2.0,  # Separación entre clases (valor alto facilita la separación)
    random_state=42 # Semilla para reproducibilidad
)

# 2. Generación de datos no linealmente separables (círculos)
X_nonlinear, y_nonlinear = make_circles(
    n_samples=100,  # Número de muestras
    noise=0.05,     # Ruido en los datos (introduce variabilidad)
    factor=0.5,     # Relación entre radios de los círculos (interno/externo)
    random_state=42 # Semilla para reproducibilidad
)

# Perceptrón mejorado para clasificación binaria
def train_perceptron(X, y, lr=0.1, epochs=100):
    """
    Entrena un perceptrón para separar datos linealmente separables.
    Parámetros:
        X: Matriz de características (n_samples, n_features)
        y: Etiquetas (-1 o 1)
        lr: Tasa de aprendizaje (learning rate)
        epochs: Número de iteraciones (épocas)
    Retorna:
        weights: Pesos ajustados del modelo
        bias: Sesgo ajustado del modelo
    """
    # Inicialización de pesos y sesgo
    weights = np.random.randn(X.shape[1]) * 0.01  # Pesos iniciales pequeños (aleatorios)
    bias = 0  # Sesgo inicial (valor escalar)

    # Entrenamiento del perceptrón
    for _ in range(epochs):  # Iterar por el número de épocas
        z = X @ weights + bias  # Producto punto entre X y los pesos, más el sesgo
        y_pred = np.where(z >= 0, 1, -1)  # Predicción binaria: 1 si z >= 0, -1 en caso contrario
        errors = y - y_pred  # Diferencia entre etiquetas reales y predicciones
        weights += lr * X.T @ errors / len(X)  # Actualización de pesos usando gradiente
        bias += lr * np.mean(errors)  # Actualización del sesgo como promedio de errores

    return weights, bias  # Retornar los pesos y sesgo ajustados

# Escalado de datos para normalizar características
X_linear_scaled = (X_linear - X_linear.mean(axis=0)) / X_linear.std(axis=0)  # Normalización de datos
w_linear, b_linear = train_perceptron(X_linear_scaled, y_linear * 2 - 1)  # Ajustar etiquetas a -1 y 1

X_nonlinear_scaled = (X_nonlinear - X_nonlinear.mean(axis=0)) / X_nonlinear.std(axis=0)  # Normalización
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
        ax: Ejes de matplotlib donde se graficará
        title: Título del gráfico
    """
    # Definir límites del gráfico
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # Límites para el eje X
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # Límites para el eje Y

    # Crear una malla de puntos para evaluar la frontera
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),  # Puntos en el eje X
        np.linspace(y_min, y_max, 100)  # Puntos en el eje Y
    )

    # Calcular valores de la frontera de decisión
    Z = xx * weights[0] + yy * weights[1] + bias  # Ecuación de la frontera
    Z = np.where(Z >= 0, 1, -1)  # Clasificación binaria en la malla

    # Graficar la frontera de decisión
    cmap = plt.cm.RdBu  # Paleta de colores para las clases
    ax.contourf(xx, yy, Z, alpha=0.4, cmap=cmap, levels=[-2, 0, 2])  # Contorno de la frontera
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=cmap, 
                          edgecolors='k', s=50)  # Puntos de datos

    # Configuración del gráfico
    ax.set_title(title, pad=15, fontsize=12)  # Título del gráfico
    ax.set_xlabel('Feature 1', labelpad=10)  # Etiqueta del eje X
    ax.set_ylabel('Feature 2', labelpad=10)  # Etiqueta del eje Y
    ax.grid(True, alpha=0.3)  # Mostrar cuadrícula con transparencia

# Visualización de los resultados
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))  # Crear dos gráficos lado a lado

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

plt.tight_layout(pad=3.0)  # Ajustar el espaciado entre gráficos
plt.show()  # Mostrar los gráficos