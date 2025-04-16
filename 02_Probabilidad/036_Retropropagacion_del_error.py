import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.preprocessing import OneHotEncoder

# --- Carga y preparación de datos ---
# Carga el conjunto de datos de dígitos (imágenes de 8x8 píxeles)
digits = load_digits()
X = digits.data / 16.0  # Normaliza los valores de píxeles al rango [0, 1]
y = digits.target.reshape(-1, 1)  # Convierte las etiquetas en un vector columna

# One-Hot Encoding para las etiquetas (necesario para la salida de la red neuronal)
encoder = OneHotEncoder(sparse_output=False)  # Configuración para evitar salida dispersa
y_onehot = encoder.fit_transform(y)  # Convierte las etiquetas en formato one-hot

# --- Definición de la red neuronal ---
class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size):
        # Inicialización de pesos y sesgos para las capas
        self.W1 = np.random.randn(input_size, hidden_size) * 0.01  # Pesos capa de entrada a oculta
        self.b1 = np.zeros((1, hidden_size))  # Sesgos capa oculta
        self.W2 = np.random.randn(hidden_size, output_size) * 0.01  # Pesos capa oculta a salida
        self.b2 = np.zeros((1, output_size))  # Sesgos capa de salida
        
    def sigmoid(self, x):
        # Función de activación sigmoide
        return 1 / (1 + np.exp(-x))
    
    def sigmoid_derivative(self, x):
        # Derivada de la función sigmoide (para retropropagación)
        return x * (1 - x)
    
    def forward(self, X):
        # Propagación hacia adelante
        self.z1 = np.dot(X, self.W1) + self.b1  # Entrada a la capa oculta
        self.a1 = self.sigmoid(self.z1)  # Activación de la capa oculta
        self.z2 = np.dot(self.a1, self.W2) + self.b2  # Entrada a la capa de salida
        self.a2 = self.sigmoid(self.z2)  # Activación de la capa de salida
        return self.a2  # Salida de la red
    
    def backward(self, X, y, lr):
        # Retropropagación del error
        m = X.shape[0]  # Número de muestras
        
        # Error en la capa de salida
        self.a2_error = y - self.a2  # Diferencia entre salida esperada y real
        self.a2_delta = self.a2_error * self.sigmoid_derivative(self.a2)  # Gradiente de salida
        
        # Error en la capa oculta
        self.a1_error = np.dot(self.a2_delta, self.W2.T)  # Propagación del error hacia atrás
        self.a1_delta = self.a1_error * self.sigmoid_derivative(self.a1)  # Gradiente de la capa oculta
        
        # Actualización de pesos y sesgos
        self.W2 += lr * np.dot(self.a1.T, self.a2_delta) / m  # Actualización de pesos capa oculta → salida
        self.b2 += lr * np.sum(self.a2_delta, axis=0) / m  # Actualización de sesgos de salida
        self.W1 += lr * np.dot(X.T, self.a1_delta) / m  # Actualización de pesos entrada → capa oculta
        self.b1 += lr * np.sum(self.a1_delta, axis=0) / m  # Actualización de sesgos de capa oculta
    
    def train(self, X, y, epochs, lr):
        # Entrenamiento de la red neuronal
        loss_history = []  # Para almacenar la pérdida en cada época
        for epoch in range(epochs):
            self.forward(X)  # Propagación hacia adelante
            self.backward(X, y, lr)  # Retropropagación y actualización de pesos
            loss = np.mean(np.square(y - self.a2))  # Cálculo del error cuadrático medio
            loss_history.append(loss)  # Almacena la pérdida
            if epoch % 500 == 0:  # Muestra el progreso cada 500 épocas
                print(f"Epoch {epoch}, Loss: {loss:.4f}")
        return loss_history  # Devuelve el historial de pérdidas

# --- Entrenamiento ---
# Inicializa la red neuronal con 64 entradas, 32 neuronas ocultas y 10 salidas
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
    # Realiza predicciones utilizando el modelo entrenado
    return np.argmax(model.forward(X), axis=1)

# Ejemplo de predicción
sample_idx = 42  # Índice de la muestra a predecir
plt.figure(figsize=(3, 3))
plt.imshow(X[sample_idx].reshape(8, 8), cmap='gray')  # Muestra la imagen de entrada
plt.title(f"Predicción: {predict([X[sample_idx]], nn)[0]}")  # Predicción del modelo
plt.axis('off')
plt.show()