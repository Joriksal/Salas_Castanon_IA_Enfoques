import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

# =============================================
# Configuración común
# =============================================
# Configuración inicial: semilla para reproducibilidad, tasa de aprendizaje y épocas
np.random.seed(42)
lr = 0.01  # Tasa de aprendizaje
epochs = 100  # Número de iteraciones

# Generar datos de clasificación binaria (linealmente separables)
X, y = make_classification(n_samples=100, n_features=2, n_redundant=0, 
                          n_classes=2, class_sep=1.5, random_state=42)
y = np.where(y == 0, -1, 1)  # Convertir etiquetas a [-1, 1] para ADALINE/MADALINE

# =============================================
# 1. Perceptrón
# =============================================
# Implementación del algoritmo Perceptrón
def perceptron(X, y, lr, epochs):
    # Inicializar pesos y sesgo
    weights = np.random.randn(X.shape[1])
    bias = 0
    
    # Entrenamiento
    for _ in range(epochs):
        for i in range(len(X)):
            # Calcular salida (z) y predicción
            z = np.dot(X[i], weights) + bias
            y_pred = 1 if z >= 0 else -1
            # Actualizar pesos y sesgo si hay error
            error = y[i] - y_pred
            weights += lr * error * X[i]
            bias += lr * error
    
    return weights, bias  # Retornar pesos y sesgo entrenados

# =============================================
# 2. ADALINE
# =============================================
# Implementación del algoritmo ADALINE
def adaline(X, y, lr, epochs):
    # Inicializar pesos y sesgo
    weights = np.random.randn(X.shape[1])
    bias = 0
    
    # Entrenamiento
    for _ in range(epochs):
        for i in range(len(X)):
            # Calcular salida (z) y error
            z = np.dot(X[i], weights) + bias
            error = y[i] - z
            # Actualizar pesos y sesgo usando el error
            weights += lr * error * X[i]
            bias += lr * error
    
    return weights, bias  # Retornar pesos y sesgo entrenados

# =============================================
# 3. MADALINE (2 ADALINEs + OR)
# =============================================
# Implementación del algoritmo MADALINE
def madaline(X, y, lr, epochs):
    # Inicializar pesos y sesgos para las dos ADALINEs en la capa oculta
    weights1 = np.random.randn(X.shape[1])
    weights2 = np.random.randn(X.shape[1])
    bias1, bias2 = 0, 0
    
    # Inicializar pesos y sesgo para la capa de salida (OR)
    weights_out = np.random.randn(2)
    bias_out = 0
    
    # Entrenamiento
    for _ in range(epochs):
        for i in range(len(X)):
            # Forward pass: calcular salidas de las ADALINEs y la capa de salida
            z1 = np.dot(X[i], weights1) + bias1
            z2 = np.dot(X[i], weights2) + bias2
            a1 = 1 if z1 >= 0 else -1
            a2 = 1 if z2 >= 0 else -1
            z_out = np.dot([a1, a2], weights_out) + bias_out
            y_pred = 1 if z_out >= 0 else -1
            
            # Backpropagation: ajustar pesos y sesgos si hay error
            error = y[i] - y_pred
            if error != 0:
                # Ajustar la ADALINE con activación más cercana a 0
                if abs(z1) < abs(z2):
                    weights1 += lr * (y[i] - z1) * X[i]
                    bias1 += lr * (y[i] - z1)
                else:
                    weights2 += lr * (y[i] - z2) * X[i]
                    bias2 += lr * (y[i] - z2)
    
    # Retornar pesos y sesgos de las ADALINEs y la capa de salida
    return (weights1, bias1), (weights2, bias2), (weights_out, bias_out)

# =============================================
# Función de visualización
# =============================================
# Función para graficar la frontera de decisión
def plot_decision_boundary(X, y, model, title, ax):
    # Crear un grid de puntos para evaluar el modelo
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),
                        np.linspace(y_min, y_max, 100))
    
    # Predecir para cada punto del grid
    if len(model) == 3:  # Caso MADALINE
        (w1, b1), (w2, b2), (w_out, b_out) = model
        Z = []
        for point in np.c_[xx.ravel(), yy.ravel()]:
            a1 = 1 if np.dot(point, w1) + b1 >= 0 else -1
            a2 = 1 if np.dot(point, w2) + b2 >= 0 else -1
            z_out = np.dot([a1, a2], w_out) + b_out
            Z.append(1 if z_out >= 0 else -1)
        Z = np.array(Z).reshape(xx.shape)
    else:  # Caso Perceptrón/ADALINE
        w, b = model
        Z = np.array([1 if np.dot(point, w) + b >= 0 else -1 
                     for point in np.c_[xx.ravel(), yy.ravel()]])
        Z = Z.reshape(xx.shape)
    
    # Graficar la frontera de decisión y los puntos de datos
    ax.contourf(xx, yy, Z, alpha=0.3, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolors='k')
    ax.set_title(title)

# =============================================
# Entrenamiento y Visualización
# =============================================
# Crear una figura con 3 subgráficos
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# Entrenar y mostrar Perceptrón
p_weights, p_bias = perceptron(X, y, lr, epochs)
plot_decision_boundary(X, y, (p_weights, p_bias), "Perceptrón", axes[0])

# Entrenar y mostrar ADALINE
a_weights, a_bias = adaline(X, y, lr, epochs)
plot_decision_boundary(X, y, (a_weights, a_bias), "ADALINE", axes[1])

# Entrenar y mostrar MADALINE
m_adaline1, m_adaline2, m_output = madaline(X, y, lr, epochs)
plot_decision_boundary(X, y, (m_adaline1, m_adaline2, m_output), "MADALINE", axes[2])

# Ajustar diseño y mostrar gráficos
plt.tight_layout()
plt.show()