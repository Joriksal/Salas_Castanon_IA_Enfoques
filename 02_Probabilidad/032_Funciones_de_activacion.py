import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons

# 1. Generar datos no lineales (en forma de luna)
# Se generan 300 puntos con ruido, distribuidos en dos clases con forma de luna.
X, y = make_moons(n_samples=300, noise=0.2, random_state=42)

# 2. Definir funciones de activación
# Estas funciones se utilizan para introducir no linealidad en la red neuronal.

# Función sigmoide: transforma valores en el rango (0, 1)
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Función ReLU: devuelve el valor original si es positivo, o 0 si es negativo
def relu(x):
    return np.maximum(0, x)

# Función tanh: transforma valores en el rango (-1, 1)
def tanh(x):
    return np.tanh(x)

# Función Leaky ReLU: similar a ReLU, pero permite un pequeño gradiente para valores negativos
def leaky_relu(x, alpha=0.01):
    return np.where(x > 0, x, alpha * x)

# 3. Arquitectura de red neuronal
# Función para entrenar una red neuronal simple con una capa oculta y una capa de salida.
def train(X, y, activation_fn, epochs=1000, lr=0.01):
    # Inicialización de pesos con valores aleatorios
    np.random.seed(42)
    W1 = np.random.randn(2, 4)  # Pesos de la capa oculta (2 entradas → 4 neuronas)
    W2 = np.random.randn(4, 1)  # Pesos de la capa de salida (4 → 1 salida)
    
    losses = []  # Lista para almacenar la pérdida en cada época
    for epoch in range(epochs):
        # Forward pass: cálculo de las salidas de cada capa
        z1 = X @ W1  # Producto punto entre entradas y pesos de la capa oculta
        a1 = activation_fn(z1)  # Aplicar función de activación en la capa oculta
        z2 = a1 @ W2  # Producto punto entre salidas de la capa oculta y pesos de la capa de salida
        a2 = sigmoid(z2)  # Salida final con función sigmoide
        
        # Cálculo de la pérdida (entropía cruzada binaria)
        loss = -np.mean(y * np.log(a2 + 1e-9) + (1 - y) * np.log(1 - a2 + 1e-9))
        losses.append(loss)
        
        # Backpropagation: cálculo de gradientes para actualizar los pesos
        dz2 = a2 - y.reshape(-1, 1)  # Gradiente de la salida
        dW2 = a1.T @ dz2  # Gradiente de los pesos de la capa de salida
        da1 = dz2 @ W2.T  # Gradiente de las activaciones de la capa oculta
        
        # Derivada según la función de activación utilizada
        if activation_fn.__name__ == "relu":
            dz1 = da1 * (z1 > 0)  # Derivada de ReLU
        elif activation_fn.__name__ == "sigmoid":
            dz1 = da1 * (a1 * (1 - a1))  # Derivada de sigmoide
        elif activation_fn.__name__ == "tanh":
            dz1 = da1 * (1 - a1**2)  # Derivada de tanh
        elif activation_fn.__name__ == "leaky_relu":
            dz1 = da1 * np.where(z1 > 0, 1, 0.01)  # Derivada de Leaky ReLU
        
        dW1 = X.T @ dz1  # Gradiente de los pesos de la capa oculta
        
        # Actualización de los pesos usando gradiente descendente
        W1 -= lr * dW1
        W2 -= lr * dW2
    
    return W1, W2, losses  # Retornar los pesos entrenados y las pérdidas

# 4. Entrenar y comparar funciones de activación
# Se entrena la red neuronal con cada función de activación y se almacenan los resultados.
activation_fns = [sigmoid, relu, tanh, leaky_relu]
results = {}

for fn in activation_fns:
    W1, W2, losses = train(X, y, fn)  # Entrenar la red con la función de activación actual
    results[fn.__name__] = {
        "weights": (W1, W2),  # Pesos entrenados
        "losses": losses  # Pérdidas durante el entrenamiento
    }

# 5. Visualización de resultados
plt.figure(figsize=(15, 10))

# Gráfico de pérdidas (subplot 1)
# Se grafican las pérdidas para cada función de activación a lo largo de las épocas.
plt.subplot(2, 2, 1)
for fn_name in results:
    plt.plot(results[fn_name]["losses"], label=fn_name)
plt.xlabel("Época")
plt.ylabel("Pérdida")
plt.legend()
plt.title("Comparación de Pérdidas")

# Función para graficar las fronteras de decisión
def plot_decision_boundary(X, y, W1, W2, activation_fn, ax, title):
    """
    Función para graficar las fronteras de decisión de un modelo de red neuronal.
    Parámetros:
    - X: Datos de entrada (matriz de características).
    - y: Etiquetas de las clases (0 o 1).
    - W1: Pesos de la capa oculta (matriz de pesos entre entrada y capa oculta).
    - W2: Pesos de la capa de salida (matriz de pesos entre capa oculta y salida).
    - activation_fn: Función de activación utilizada en la capa oculta.
    - ax: Objeto de ejes de Matplotlib donde se graficará.
    - title: Título del gráfico.
    """
    
    # Crear una malla de puntos en el espacio de entrada para evaluar el modelo
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # Límites del eje x
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # Límites del eje y
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100),  # Crear una cuadrícula de puntos
                         np.linspace(y_min, y_max, 100))  # 100 puntos en cada eje
    
    # Calcular las salidas de la red para cada punto de la malla
    Z = np.c_[xx.ravel(), yy.ravel()]  # Combinar las coordenadas de la malla en un arreglo bidimensional
    z1 = Z @ W1  # Producto punto entre los puntos de la malla y los pesos de la capa oculta
    a1 = activation_fn(z1)  # Aplicar la función de activación en la capa oculta
    z2 = a1 @ W2  # Producto punto entre las salidas de la capa oculta y los pesos de la capa de salida
    a2 = sigmoid(z2)  # Aplicar la función sigmoide para obtener la probabilidad de la clase
    Z = a2.reshape(xx.shape)  # Reorganizar las salidas en la forma de la malla original
    
    # Graficar las fronteras de decisión
    ax.contourf(xx, yy, Z > 0.5, alpha=0.3, cmap=plt.cm.RdYlBu)  # Colorear las regiones según la clase predicha
    ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k', cmap=plt.cm.RdYlBu)  # Graficar los puntos de datos originales
    ax.set_title(title)  # Establecer el título del gráfico

# Gráficos de fronteras de decisión (subplots 2-4)
fn_names = list(results.keys())
for idx, pos in enumerate([2, 3, 4]):
    ax = plt.subplot(2, 2, pos)
    fn_name = fn_names[idx]
    W1, W2 = results[fn_name]["weights"]
    plot_decision_boundary(X, y, W1, W2, globals()[fn_name], ax, 
                         f"{fn_name}")

# Ajustar diseño y mostrar gráficos
plt.tight_layout()
plt.show()