# --- Importación de librerías necesarias ---
import numpy as np  # Librería para operaciones matemáticas y manejo de matrices
import matplotlib.pyplot as plt  # Librería para visualización de datos y gráficos
from sklearn.datasets import load_digits  # Conjunto de datos de dígitos predefinido en scikit-learn
from sklearn.preprocessing import OneHotEncoder  # Herramienta para codificación One-Hot de etiquetas

# --- Carga y preparación de datos ---
# Carga el conjunto de datos de dígitos (imágenes de 8x8 píxeles)
digits = load_digits()  # Carga el dataset de dígitos
X = digits.data / 16.0  # Normaliza los valores de píxeles al rango [0, 1] dividiendo entre 16
y = digits.target.reshape(-1, 1)  # Convierte las etiquetas en un vector columna para facilitar el procesamiento

# One-Hot Encoding para las etiquetas (necesario para la salida de la red neuronal)
encoder = OneHotEncoder(sparse_output=False)  # Configuración para evitar salida dispersa (matriz densa)
y_onehot = encoder.fit_transform(y)  # Convierte las etiquetas en formato One-Hot (matriz binaria)

# --- Definición de la red neuronal ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        """
        Inicializa la red neuronal con pesos y sesgos aleatorios.
        Parámetros:
        - input_size: Número de neuronas en la capa de entrada (dimensión de los datos de entrada)
        - hidden_size: Número de neuronas en la capa oculta
        - output_size: Número de neuronas en la capa de salida (dimensión de las etiquetas codificadas One-Hot)
        """
        # Pesos entre la capa de entrada y la capa oculta, inicializados aleatoriamente
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01
        # Sesgos para la capa oculta, inicializados en ceros
        self.b1 = np.zeros((1, hidden_size))
        # Pesos entre la capa oculta y la capa de salida, inicializados aleatoriamente
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01
        # Sesgos para la capa de salida, inicializados en ceros
        self.b2 = np.zeros((1, output_size))
        
    def sigmoid(self, x):
        """
        Función de activación sigmoide.
        Convierte cualquier valor en un rango entre 0 y 1.
        """
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        """
        Derivada de la función sigmoide.
        Se utiliza para calcular los gradientes durante la retropropagación.
        """
        return x * (1 - x)
    
    def forward(self, X):
        """
        Realiza la propagación hacia adelante (forward pass).
        Calcula las salidas de cada capa de la red neuronal.
        Parámetros:
        - X: Datos de entrada
        Retorna:
        - a2: Salida final de la red neuronal
        """
        # Cálculo de la entrada a la capa oculta
        self.z1 = np.dot(X, self.W1) + self.b1
        # Aplicación de la función de activación sigmoide en la capa oculta
        self.a1 = self.sigmoid(self.z1)
        # Cálculo de la entrada a la capa de salida
        self.z2 = np.dot(self.a1, self.W2) + self.b2
        # Aplicación de la función de activación sigmoide en la capa de salida
        self.a2 = self.sigmoid(self.z2)
        return self.a2  # Retorna la salida final de la red
    
    def backward(self, X, y, lr):
        """
        Realiza la retropropagación del error (backward pass).
        Calcula los gradientes y actualiza los pesos y sesgos de la red.
        Parámetros:
        - X: Datos de entrada
        - y: Etiquetas esperadas (en formato One-Hot)
        - lr: Tasa de aprendizaje
        """
        m = X.shape[0]  # Número de muestras en el conjunto de datos
        
        # Error en la capa de salida (diferencia entre salida esperada y real)
        self.a2_error = y - self.a2
        # Gradiente de la capa de salida
        self.a2_delta = self.a2_error * self.sigmoid_derivative(self.a2)
        
        # Error en la capa oculta (propagación del error hacia atrás)
        self.a1_error = np.dot(self.a2_delta, self.W2.T)
        # Gradiente de la capa oculta
        self.a1_delta = self.a1_error * self.sigmoid_derivative(self.a1)
        
        # Actualización de pesos y sesgos utilizando los gradientes calculados
        self.W2 += lr * np.dot(self.a1.T, self.a2_delta) / m
        self.b2 += lr * np.sum(self.a2_delta, axis=0) / m
        self.W1 += lr * np.dot(X.T, self.a1_delta) / m
        self.b1 += lr * np.sum(self.a1_delta, axis=0) / m
    
    def train(self, X, y, epochs, lr):
        """
        Entrena la red neuronal utilizando propagación hacia adelante y retropropagación.
        Parámetros:
        - X: Datos de entrada
        - y: Etiquetas esperadas (en formato One-Hot)
        - epochs: Número de épocas de entrenamiento
        - lr: Tasa de aprendizaje
        Retorna:
        - loss_history: Historial de pérdidas durante el entrenamiento
        """
        loss_history = []  # Lista para almacenar la pérdida en cada época
        for epoch in range(epochs):
            self.forward(X)  # Propagación hacia adelante
            self.backward(X, y, lr)  # Retropropagación y actualización de pesos
            # Cálculo del error cuadrático medio (MSE)
            loss = np.mean(np.square(y - self.a2))
            loss_history.append(loss)  # Almacena la pérdida
            if epoch % 500 == 0:  # Muestra el progreso cada 500 épocas
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss_history  # Devuelve el historial de pérdidas

# --- Entrenamiento ---
# Inicializa la red neuronal con 64 entradas (8x8 píxeles), 32 neuronas ocultas y 10 salidas (dígitos 0-9)
nn = NeuralNetwork(input_size=64, hidden_size=32, output_size=10)
# Entrena la red con 3000 épocas y una tasa de aprendizaje de 0.1
loss_history = nn.train(X, y_onehot, epochs=3000, lr=0.1)

# --- Visualización ---
plt.figure(figsize=(15, 5))

# 1. Evolución de la pérdida durante el entrenamiento
plt.subplot(1, 2, 1)
plt.plot(loss_history)
plt.title("Retropropagación: Pérdida durante el entrenamiento")
plt.xlabel("Época")
plt.ylabel("Error Cuadrático Medio")

# 2. Visualización de gradientes en la capa oculta
plt.subplot(1, 2, 2)
gradients = np.abs(nn.a1_delta)[0].reshape(8, 4)  # Gradientes de la primera muestra
plt.imshow(gradients, cmap='hot', interpolation='nearest')
plt.colorbar()
plt.title("Mapa de Gradientes (Capa Oculta)")
plt.xlabel("Neuronas Oculta")
plt.ylabel("Input Features")

plt.tight_layout()
plt.show()

# --- Predicción ---
def predict(X, model):
    """
    Realiza predicciones utilizando el modelo entrenado.
    Parámetros:
    - X: Datos de entrada
    - model: Modelo entrenado
    Retorna:
    - Predicciones (índices de las clases con mayor probabilidad)
    """
    return np.argmax(model.forward(X), axis=1)

# Ejemplo de predicción
sample_idx = 42  # Índice de la muestra a predecir
plt.figure(figsize=(3, 3))
plt.imshow(X[sample_idx].reshape(8, 8), cmap='gray')  # Muestra la imagen de entrada
plt.title(f"Predicción: {predict([X[sample_idx]], nn)[0]}")  # Predicción del modelo
plt.axis('off')
plt.show()