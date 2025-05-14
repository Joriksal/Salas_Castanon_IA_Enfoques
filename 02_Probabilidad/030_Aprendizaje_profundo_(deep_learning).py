# Importar las librerías necesarias
import numpy as np  # Librería para cálculos numéricos y manejo de arreglos
import matplotlib.pyplot as plt  # Librería para visualización de datos y gráficos

# Generar datos sintéticos para el modelo
np.random.seed(42)  # Fijar la semilla para garantizar reproducibilidad en los resultados
X = np.linspace(-3, 3, 1000)  # Crear 1000 puntos equidistantes entre -3 y 3 (característica X)
y = X + np.random.normal(0, 0.5 * np.abs(X), len(X))  # Generar valores de salida (y) con ruido proporcional a |X|

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, y_train = X[:800], y[:800]  # Usar los primeros 800 puntos para el entrenamiento
X_test, y_test = X[800:], y[800:]  # Usar los últimos 200 puntos para la prueba

# Normalizar los datos para mejorar la estabilidad del entrenamiento
X_mean, X_std = np.mean(X_train), np.std(X_train)  # Calcular la media y desviación estándar del conjunto de entrenamiento
X_train = (X_train - X_mean) / X_std  # Normalizar los datos de entrenamiento
X_test = (X_test - X_mean) / X_std  # Normalizar los datos de prueba usando la misma media y desviación estándar

# Definir los hiperparámetros de la red neuronal
input_dim = 1  # Dimensión de entrada (1 característica: X)
hidden_dim = 64  # Número de neuronas en la capa oculta
learning_rate = 0.01  # Tasa de aprendizaje para la optimización
epochs = 2000  # Número de épocas (iteraciones) para entrenar la red

# Función para inicializar los pesos y sesgos de las capas
def init_weights(dim_in, dim_out):
    """
    Inicializa los pesos y sesgos de una capa de la red neuronal.
    - Los pesos se inicializan con una distribución normal escalada (Inicialización He).
    - Los sesgos se inicializan en cero.
    """
    W = np.random.randn(dim_in, dim_out) * np.sqrt(2. / dim_in)  # Pesos inicializados con He initialization
    b = np.zeros(dim_out)  # Sesgos inicializados en cero
    return W, b

# Inicializar los pesos y sesgos para las capas de la red
W1, b1 = init_weights(input_dim, hidden_dim)  # Pesos y sesgos para la capa oculta
W2_mu, b2_mu = init_weights(hidden_dim, 1)  # Pesos y sesgos para la salida de la media (μ)
W2_logvar, b2_logvar = init_weights(hidden_dim, 1)  # Pesos y sesgos para la salida del logaritmo de la varianza (log(σ²))

# Definir la función de activación ReLU
def relu(x):
    """
    Aplica la función de activación ReLU (Rectified Linear Unit).
    - ReLU devuelve el valor original si es positivo, o 0 si es negativo.
    """
    return np.maximum(0, x)

# Implementar la propagación hacia adelante (forward pass)
def forward(X):
    """
    Realiza la propagación hacia adelante en la red neuronal.
    - Calcula las salidas de la capa oculta y las salidas finales (μ y log(σ²)).
    """
    hidden = relu(X @ W1 + b1)  # Cálculo de la capa oculta con activación ReLU
    mu = hidden @ W2_mu + b2_mu  # Salida para la media (μ)
    log_var = hidden @ W2_logvar + b2_logvar  # Salida para el logaritmo de la varianza (log(σ²))
    return mu, log_var

# Definir la función de pérdida: Log-Likelihood Negativo (NLL)
def nll_loss(y, mu, log_var):
    """
    Calcula la pérdida de log-verosimilitud negativa (NLL).
    - Penaliza predicciones que se desvíen de los valores reales y mide la incertidumbre.
    """
    return np.mean(0.5 * log_var + 0.5 * ((y - mu)**2) / np.exp(log_var))

# Entrenar la red neuronal
loss_history = []  # Lista para almacenar la pérdida en cada época
for epoch in range(epochs):
    # Propagación hacia adelante
    mu, log_var = forward(X_train.reshape(-1, 1))  # Calcular las salidas de la red para los datos de entrenamiento
    loss = nll_loss(y_train.reshape(-1, 1), mu, log_var)  # Calcular la pérdida
    loss_history.append(loss)  # Guardar la pérdida en la lista
    
    # Backpropagation: cálculo de gradientes
    grad_mu = (mu - y_train.reshape(-1, 1)) / np.exp(log_var)  # Gradiente respecto a μ
    grad_log_var = 0.5 * (1 - ((y_train.reshape(-1, 1) - mu)**2 / np.exp(log_var)))  # Gradiente respecto a log(σ²)
    
    # Gradientes para la capa oculta
    grad_hidden_mu = grad_mu @ W2_mu.T  # Gradiente de la capa oculta respecto a μ
    grad_hidden_logvar = grad_log_var @ W2_logvar.T  # Gradiente de la capa oculta respecto a log(σ²)
    grad_hidden = (grad_hidden_mu + grad_hidden_logvar) * (X_train.reshape(-1, 1) @ W1 + b1 > 0)  # Derivada de ReLU
    
    # Actualización de los pesos y sesgos usando gradientes descendentes
    W2_mu -= learning_rate * (relu(X_train.reshape(-1, 1) @ W1 + b1).T @ grad_mu) / len(X_train)
    b2_mu -= learning_rate * np.sum(grad_mu, axis=0) / len(X_train)
    W2_logvar -= learning_rate * (relu(X_train.reshape(-1, 1) @ W1 + b1).T @ grad_log_var) / len(X_train)
    b2_logvar -= learning_rate * np.sum(grad_log_var, axis=0) / len(X_train)
    W1 -= learning_rate * X_train.reshape(-1, 1).T @ grad_hidden / len(X_train)
    b1 -= learning_rate * np.sum(grad_hidden, axis=0) / len(X_train)

# Realizar predicciones en el conjunto de prueba
mu_test, log_var_test = forward(X_test.reshape(-1, 1))  # Propagación hacia adelante en los datos de prueba
sigma_test = np.exp(0.5 * log_var_test)  # Calcular la desviación estándar (σ) a partir de log(σ²)

# Visualizar los resultados
plt.figure(figsize=(12, 5))  # Configurar el tamaño de la figura
plt.scatter(X_train, y_train, alpha=0.3, label='Train data')  # Graficar los datos de entrenamiento
plt.scatter(X_test, y_test, alpha=0.3, label='Test data')  # Graficar los datos de prueba
plt.plot(X_test, mu_test, color='red', label='Predicción (μ)')  # Graficar la predicción de la media
plt.fill_between(
    X_test.flatten(),
    (mu_test - 1.96 * sigma_test).flatten(),  # Límite inferior de la banda de incertidumbre (μ - 1.96σ)
    (mu_test + 1.96 * sigma_test).flatten(),  # Límite superior de la banda de incertidumbre (μ + 1.96σ)
    alpha=0.2, color='red', label='Incertidumbre (±1.96σ)'  # Banda de incertidumbre
)
plt.xlabel('x (normalizado)')  # Etiqueta del eje x
plt.ylabel('y')  # Etiqueta del eje y
plt.legend()  # Mostrar la leyenda
plt.title('Regresión Probabilística con NumPy')  # Título del gráfico
plt.show()  # Mostrar el gráfico