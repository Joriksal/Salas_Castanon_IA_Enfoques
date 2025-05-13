# =============================================
# Importación de librerías
# =============================================
# numpy: Librería para trabajar con arreglos y realizar operaciones matemáticas de manera eficiente.
import numpy as np

# matplotlib.pyplot: Librería para crear gráficos y visualizaciones.
import matplotlib.pyplot as plt

# sklearn.datasets.make_classification: Función para generar datos de clasificación sintéticos.
from sklearn.datasets import make_classification

# =============================================
# Configuración común
# =============================================
# Configuración inicial: semilla para reproducibilidad, tasa de aprendizaje y épocas
np.random.seed(42)  # Fija la semilla para que los resultados sean reproducibles.
lr = 0.01  # Tasa de aprendizaje, controla el tamaño de los pasos en la actualización de pesos.
epochs = 100  # Número de iteraciones para entrenar los modelos.

# Generar datos de clasificación binaria (linealmente separables)
X, y = make_classification(
    n_samples=100,  # Número de muestras (puntos de datos).
    n_features=2,  # Número de características (dimensiones).
    n_redundant=0,  # Características redundantes (no se generan).
    n_classes=2,  # Número de clases (binario: 0 y 1).
    class_sep=1.5,  # Separación entre las clases (mayor valor = más separadas).
    random_state=42  # Semilla para reproducibilidad.
)

# Convertir etiquetas de clase (0 y 1) a (-1 y 1) para que sean compatibles con ADALINE/MADALINE.
y = np.where(y == 0, -1, 1)

# =============================================
# 1. Perceptrón
# =============================================
# Implementación del algoritmo Perceptrón
def perceptron(X, y, lr, epochs):
    """
    Algoritmo Perceptrón para clasificación binaria.
    - X: Datos de entrada (características).
    - y: Etiquetas de clase (-1 o 1).
    - lr: Tasa de aprendizaje.
    - epochs: Número de iteraciones.
    """
    # Inicializar pesos (aleatorios) y sesgo (bias).
    weights = np.random.randn(X.shape[1])  # Pesos para cada característica.
    bias = 0  # Sesgo inicial.

    # Entrenamiento
    for _ in range(epochs):  # Repetir por el número de épocas.
        for i in range(len(X)):  # Iterar sobre cada muestra.
            # Calcular salida (z) y predicción.
            z = np.dot(X[i], weights) + bias  # Producto punto entre entrada y pesos + sesgo.
            y_pred = 1 if z >= 0 else -1  # Clasificación basada en el signo de z.

            # Actualizar pesos y sesgo si hay error.
            error = y[i] - y_pred  # Diferencia entre etiqueta real y predicción.
            weights += lr * error * X[i]  # Ajustar pesos.
            bias += lr * error  # Ajustar sesgo.

    return weights, bias  # Retornar pesos y sesgo entrenados.

# =============================================
# 2. ADALINE
# =============================================
# Implementación del algoritmo ADALINE
def adaline(X, y, lr, epochs):
    """
    Algoritmo ADALINE (Adaptative Linear Neuron).
    - X: Datos de entrada.
    - y: Etiquetas de clase.
    - lr: Tasa de aprendizaje.
    - epochs: Número de iteraciones.
    """
    # Inicializar pesos y sesgo.
    weights = np.random.randn(X.shape[1])  # Pesos iniciales.
    bias = 0  # Sesgo inicial.

    # Entrenamiento
    for _ in range(epochs):  # Repetir por el número de épocas.
        for i in range(len(X)):  # Iterar sobre cada muestra.
            # Calcular salida (z) y error.
            z = np.dot(X[i], weights) + bias  # Producto punto + sesgo.
            error = y[i] - z  # Error continuo (no binario).

            # Actualizar pesos y sesgo usando el error.
            weights += lr * error * X[i]  # Ajustar pesos.
            bias += lr * error  # Ajustar sesgo.

    return weights, bias  # Retornar pesos y sesgo entrenados.

# =============================================
# 3. MADALINE (2 ADALINEs + OR)
# =============================================
# Implementación del algoritmo MADALINE
def madaline(X, y, lr, epochs):
    """
    Algoritmo MADALINE (Multiple ADALINE).
    - X: Datos de entrada.
    - y: Etiquetas de clase.
    - lr: Tasa de aprendizaje.
    - epochs: Número de iteraciones.
    """
    # Inicializar pesos y sesgos para las dos ADALINEs en la capa oculta.
    weights1 = np.random.randn(X.shape[1])  # Pesos para la primera ADALINE.
    weights2 = np.random.randn(X.shape[1])  # Pesos para la segunda ADALINE.
    bias1, bias2 = 0, 0  # Sesgos iniciales.

    # Inicializar pesos y sesgo para la capa de salida (OR).
    weights_out = np.random.randn(2)  # Pesos para la salida.
    bias_out = 0  # Sesgo inicial.

    # Entrenamiento
    for _ in range(epochs):  # Repetir por el número de épocas.
        for i in range(len(X)):  # Iterar sobre cada muestra.
            # Forward pass: calcular salidas de las ADALINEs y la capa de salida.
            z1 = np.dot(X[i], weights1) + bias1  # Salida de la primera ADALINE.
            z2 = np.dot(X[i], weights2) + bias2  # Salida de la segunda ADALINE.
            a1 = 1 if z1 >= 0 else -1  # Activación de la primera ADALINE.
            a2 = 1 if z2 >= 0 else -1  # Activación de la segunda ADALINE.
            z_out = np.dot([a1, a2], weights_out) + bias_out  # Salida final.
            y_pred = 1 if z_out >= 0 else -1  # Predicción final.

            # Backpropagation: ajustar pesos y sesgos si hay error.
            error = y[i] - y_pred  # Error en la salida.
            if error != 0:  # Si hay error, ajustar la ADALINE más cercana a 0.
                if abs(z1) < abs(z2):
                    weights1 += lr * (y[i] - z1) * X[i]
                    bias1 += lr * (y[i] - z1)
                else:
                    weights2 += lr * (y[i] - z2) * X[i]
                    bias2 += lr * (y[i] - z2)

    # Retornar pesos y sesgos de las ADALINEs y la capa de salida.
    return (weights1, bias1), (weights2, bias2), (weights_out, bias_out)

# =============================================
# Función de visualización
# =============================================
# Función para graficar la frontera de decisión
def plot_decision_boundary(X, y, model, title, ax):
    """
    Graficar la frontera de decisión de un modelo.
    - X: Datos de entrada.
    - y: Etiquetas de clase.
    - model: Modelo entrenado (pesos y sesgos).
    - title: Título del gráfico.
    - ax: Subgráfico donde se dibujará.
    """
    # Crear un grid de puntos para evaluar el modelo.
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 100),  # Puntos en el eje x.
        np.linspace(y_min, y_max, 100)  # Puntos en el eje y.
    )

    # Predecir para cada punto del grid.
    if len(model) == 3:  # Caso MADALINE.
        (w1, b1), (w2, b2), (w_out, b_out) = model
        Z = []
        for point in np.c_[xx.ravel(), yy.ravel()]:  # Iterar sobre cada punto del grid.
            a1 = 1 if np.dot(point, w1) + b1 >= 0 else -1
            a2 = 1 if np.dot(point, w2) + b2 >= 0 else -1
            z_out = np.dot([a1, a2], w_out) + b_out
            Z.append(1 if z_out >= 0 else -1)
        Z = np.array(Z).reshape(xx.shape)
    else:  # Caso Perceptrón/ADALINE.
        w, b = model
        Z = np.array([
            1 if np.dot(point, w) + b >= 0 else -1
            for point in np.c_[xx.ravel(), yy.ravel()]
        ])
        Z = Z.reshape(xx.shape)

    # Graficar la frontera de decisión y los puntos de datos.
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')  # Frontera de decisión.
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')  # Puntos de datos.
    ax.set_title(title)  # Título del gráfico.

# =============================================
# Entrenamiento y Visualización
# =============================================
# Crear una figura con 3 subgráficos.
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Entrenar y mostrar Perceptrón.
p_weights, p_bias = perceptron(X, y, lr, epochs)
plot_decision_boundary(X, y, (p_weights, p_bias), "Perceptrón", axes[0])

# Entrenar y mostrar ADALINE.
a_weights, a_bias = adaline(X, y, lr, epochs)
plot_decision_boundary(X, y, (a_weights, a_bias), "ADALINE", axes[1])

# Entrenar y mostrar MADALINE.
m_adaline1, m_adaline2, m_output = madaline(X, y, lr, epochs)
plot_decision_boundary(X, y, (m_adaline1, m_adaline2, m_output), "MADALINE", axes[2])

# Ajustar diseño y mostrar gráficos.
plt.tight_layout()  # Ajustar diseño para evitar solapamientos.
plt.show()  # Mostrar los gráficos.