import numpy as np
import matplotlib.pyplot as plt

# Generar datos sintéticos
np.random.seed(42)  # Fijar la semilla para reproducibilidad
X = np.linspace(-3, 3, 1000)  # Generar 1000 puntos equidistantes entre -3 y 3
y = X + np.random.normal(0, 0.5 * np.abs(X), len(X))  # Agregar ruido proporcional a |X|

# Dividir en conjuntos de entrenamiento y prueba
X_train, y_train = X[:800], y[:800]  # Primeros 800 puntos para entrenamiento
X_test, y_test = X[800:], y[800:]  # Últimos 200 puntos para prueba

# Normalizar los datos (media 0, desviación estándar 1)
X_mean, X_std = np.mean(X_train), np.std(X_train)  # Calcular media y desviación estándar
X_train = (X_train - X_mean) / X_std  # Normalizar conjunto de entrenamiento
X_test = (X_test - X_mean) / X_std  # Normalizar conjunto de prueba

# Hiperparámetros de la red neuronal
input_dim = 1  # Dimensión de entrada (1 característica: X)
hidden_dim = 64  # Número de neuronas en la capa oculta
learning_rate = 0.01  # Tasa de aprendizaje
epochs = 2000  # Número de épocas de entrenamiento

# Inicialización de pesos y sesgos
def init_weights(dim_in, dim_out):
    """
    Inicializa los pesos con distribución normal escalada y los sesgos en cero.
    """
    W = np.random.randn(dim_in, dim_out) * np.sqrt(2. / dim_in)  # Inicialización He
    b = np.zeros(dim_out)  # Sesgos inicializados en cero
    return W, b

# Inicializar pesos para las capas
W1, b1 = init_weights(input_dim, hidden_dim)  # Capa oculta
W2_mu, b2_mu = init_weights(hidden_dim, 1)  # Salida para la media (μ)
W2_logvar, b2_logvar = init_weights(hidden_dim, 1)  # Salida para el logaritmo de la varianza (log(σ²))

# Función de activación ReLU
def relu(x):
    """
    Aplica la función ReLU (Rectified Linear Unit).
    """
    return np.maximum(0, x)

# Forward pass (propagación hacia adelante)
def forward(X):
    """
    Realiza la propagación hacia adelante en la red neuronal.
    """
    hidden = relu(X @ W1 + b1)  # Capa oculta con activación ReLU
    mu = hidden @ W2_mu + b2_mu  # Salida para la media (μ)
    log_var = hidden @ W2_logvar + b2_logvar  # Salida para log(σ²)
    return mu, log_var

# Función de pérdida: Log-Likelihood Negativo (NLL)
def nll_loss(y, mu, log_var):
    """
    Calcula la pérdida de log-verosimilitud negativa.
    """
    return np.mean(0.5 * log_var + 0.5 * ((y - mu)**2) / np.exp(log_var))

# Entrenamiento de la red neuronal
loss_history = []  # Para almacenar la pérdida en cada época
for epoch in range(epochs):
    # Forward pass
    mu, log_var = forward(X_train.reshape(-1, 1))  # Propagación hacia adelante
    loss = nll_loss(y_train.reshape(-1, 1), mu, log_var)  # Calcular pérdida
    loss_history.append(loss)  # Guardar pérdida
    
    # Backpropagation (cálculo de gradientes)
    grad_mu = (mu - y_train.reshape(-1, 1)) / np.exp(log_var)  # Gradiente respecto a μ
    grad_log_var = 0.5 * (1 - ((y_train.reshape(-1, 1) - mu)**2 / np.exp(log_var)))  # Gradiente respecto a log(σ²)
    
    # Gradientes para la capa oculta
    grad_hidden_mu = grad_mu @ W2_mu.T
    grad_hidden_logvar = grad_log_var @ W2_logvar.T
    grad_hidden = (grad_hidden_mu + grad_hidden_logvar) * (X_train.reshape(-1, 1) @ W1 + b1 > 0)  # Derivada de ReLU
    
    # Actualización de pesos y sesgos
    W2_mu -= learning_rate * (relu(X_train.reshape(-1, 1) @ W1 + b1).T @ grad_mu) / len(X_train)
    b2_mu -= learning_rate * np.sum(grad_mu, axis=0) / len(X_train)
    W2_logvar -= learning_rate * (relu(X_train.reshape(-1, 1) @ W1 + b1).T @ grad_log_var) / len(X_train)
    b2_logvar -= learning_rate * np.sum(grad_log_var, axis=0) / len(X_train)
    W1 -= learning_rate * X_train.reshape(-1, 1).T @ grad_hidden / len(X_train)
    b1 -= learning_rate * np.sum(grad_hidden, axis=0) / len(X_train)

# Predicción en el conjunto de prueba
mu_test, log_var_test = forward(X_test.reshape(-1, 1))  # Propagación hacia adelante en el conjunto de prueba
sigma_test = np.exp(0.5 * log_var_test)  # Calcular desviación estándar (σ)

# Visualización de resultados
plt.figure(figsize=(12, 5))
plt.scatter(X_train, y_train, alpha=0.3, label='Train data')  # Datos de entrenamiento
plt.scatter(X_test, y_test, alpha=0.3, label='Test data')  # Datos de prueba
plt.plot(X_test, mu_test, color='red', label='Predicción (μ)')  # Predicción de la media
plt.fill_between(
    X_test.flatten(),
    (mu_test - 1.96 * sigma_test).flatten(),  # Límite inferior (μ - 1.96σ)
    (mu_test + 1.96 * sigma_test).flatten(),  # Límite superior (μ + 1.96σ)
    alpha=0.2, color='red', label='Incertidumbre (±1.96σ)'  # Banda de incertidumbre
)
plt.xlabel('x (normalizado)')
plt.ylabel('y')
plt.legend()
plt.title('Regresión Probabilística con NumPy')
plt.show()