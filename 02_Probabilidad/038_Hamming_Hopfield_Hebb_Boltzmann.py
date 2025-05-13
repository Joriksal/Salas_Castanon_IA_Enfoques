# --- Importación de librerías ---
# numpy: Librería para trabajar con arreglos y operaciones matemáticas.
import numpy as np

# matplotlib.pyplot: Librería para generar gráficos y visualizaciones.
import matplotlib.pyplot as plt

# torch: Librería para computación numérica y aprendizaje profundo.
import torch

# torch.nn: Módulo de PyTorch que contiene herramientas para construir redes neuronales.
import torch.nn as nn

# --- 1. Red de Hamming (Clasificación) ---
class HammingNetwork:
    def __init__(self, num_patterns, pattern_size):
        """
        Inicializa una red de Hamming con una matriz de pesos de tamaño
        (num_patterns x pattern_size), donde:
        - num_patterns: Número de patrones que la red puede clasificar.
        - pattern_size: Tamaño de cada patrón.
        """
        self.weights = np.zeros((num_patterns, pattern_size))  # Matriz de pesos inicializada en ceros.
    
    def train(self, patterns):
        """
        Entrena la red copiando los patrones directamente como pesos.
        - patterns: Matriz donde cada fila es un patrón binario.
        """
        self.weights = patterns.copy()  # Copia los patrones como pesos.
    
    def predict(self, input_pattern):
        """
        Clasifica un patrón de entrada calculando su similitud con los patrones entrenados.
        - input_pattern: Patrón binario de entrada.
        Devuelve el índice del patrón más similar.
        """
        similarities = np.dot(self.weights, input_pattern)  # Producto punto para calcular similitud.
        return np.argmax(similarities)  # Índice del patrón más similar.

# --- 2. Red de Hopfield (Memoria Asociativa) ---
class HopfieldNetwork:
    def __init__(self, size):
        """
        Inicializa una red de Hopfield con una matriz de pesos de tamaño (size x size).
        - size: Número de neuronas en la red.
        """
        self.weights = np.zeros((size, size))  # Matriz de pesos inicializada en ceros.
    
    def train(self, patterns):
        """
        Entrena la red utilizando la regla de aprendizaje de Hebb.
        - patterns: Lista de patrones binarios para entrenar la red.
        """
        for p in patterns:
            self.weights += np.outer(p, p)  # Producto externo para actualizar los pesos.
        np.fill_diagonal(self.weights, 0)  # Elimina autoconexiones (diagonal en ceros).
    
    def predict(self, input_pattern, max_iter=100):
        """
        Recupera un patrón a partir de una entrada corrupta.
        - input_pattern: Patrón binario de entrada.
        - max_iter: Número máximo de iteraciones para la convergencia.
        Devuelve el patrón recuperado.
        """
        pattern = input_pattern.copy()  # Copia el patrón de entrada.
        for _ in range(max_iter):
            new_pattern = np.sign(np.dot(self.weights, pattern))  # Actualiza el patrón.
            if np.array_equal(new_pattern, pattern):  # Si converge, detiene la iteración.
                break
            pattern = new_pattern
        return pattern

# --- 3. Regla de Hebb (Aprendizaje No Supervisado) ---
def hebb_rule(input_data, output_data, lr=0.1):
    """
    Calcula los pesos utilizando la regla de Hebb.
    - input_data: Matriz de entradas.
    - output_data: Matriz de salidas deseadas.
    - lr: Tasa de aprendizaje (learning rate).
    Devuelve la matriz de pesos aprendidos.
    """
    weights = np.zeros((output_data.shape[1], input_data.shape[1]))  # Inicializa pesos en ceros.
    for x, y in zip(input_data, output_data):
        weights += lr * np.outer(y, x)  # Actualiza los pesos con el producto externo.
    return weights

# --- 4. Máquina de Boltzmann (Generativa) ---
class BoltzmannMachine(nn.Module):
    def __init__(self, num_visible, num_hidden):
        """
        Inicializa una máquina de Boltzmann con capas visibles y ocultas.
        - num_visible: Número de neuronas visibles.
        - num_hidden: Número de neuronas ocultas.
        """
        super().__init__()  # Llama al constructor de la clase base nn.Module.
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 0.1)  # Pesos inicializados aleatoriamente.
        self.b = nn.Parameter(torch.zeros(num_hidden))  # Sesgos inicializados en ceros.
    
    def forward(self, v):
        """
        Calcula las probabilidades de activación de las neuronas ocultas.
        - v: Vector de entrada (neuronas visibles).
        Devuelve las probabilidades de activación de las neuronas ocultas.
        """
        h_prob = torch.sigmoid(torch.matmul(self.W, v.t()) + self.b)  # Función sigmoide para probabilidades.
        return h_prob

# --- Datos de Ejemplo ---
# Patrones binarios para Hamming y Hopfield.
patterns = np.array([
    [1, 1, -1, -1],  # Patrón A
    [-1, -1, 1, 1]   # Patrón B
])
test_pattern = np.array([1, -1, -1, -1])  # Patrón corrupto (debería recuperar A).

# Entradas y salidas para la regla de Hebb.
X_hebb = np.array([[1, -1], [-1, 1]])  # Entradas.
Y_hebb = np.array([[1, 0], [0, 1]])  # Salidas deseadas.

# Datos simulados para la máquina de Boltzmann.
data_boltzmann = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float)

# --- Ejecución y Visualización ---
def main():
    # --- Hamming ---
    hn_hamming = HammingNetwork(num_patterns=2, pattern_size=4)  # Inicializa la red.
    hn_hamming.train(patterns)  # Entrena con los patrones binarios.
    hamming_result = hn_hamming.predict(test_pattern)  # Clasifica el patrón corrupto.
    print(f"Hamming: Patrón clasificado como {hamming_result}")

    # --- Hopfield ---
    hn_hopfield = HopfieldNetwork(size=4)  # Inicializa la red.
    hn_hopfield.train(patterns)  # Entrena con los patrones binarios.
    hopfield_result = hn_hopfield.predict(test_pattern)  # Recupera el patrón original.
    print(f"Hopfield: Patrón recuperado:\n{hopfield_result}")

    # --- Hebb ---
    hebb_weights = hebb_rule(X_hebb, Y_hebb)  # Calcula los pesos con la regla de Hebb.
    print(f"Hebb: Pesos aprendidos:\n{hebb_weights}")

    # --- Boltzmann ---
    bm = BoltzmannMachine(num_visible=3, num_hidden=2)  # Inicializa la máquina de Boltzmann.
    with torch.no_grad():  # Desactiva el cálculo de gradientes (no es necesario aquí).
        sample = torch.tensor([1, 0, 1], dtype=torch.float)  # Entrada de ejemplo.
        boltzmann_result = bm(sample).numpy()  # Calcula las probabilidades de las neuronas ocultas.
    print(f"Boltzmann: Probabilidad de neuronas ocultas:\n{boltzmann_result}")

    # --- Visualización ---
    plt.figure(figsize=(10, 4))  # Configura el tamaño de la figura.
    plt.subplot(1, 3, 1)  # Primer subgráfico.
    plt.imshow([test_pattern], cmap='coolwarm', aspect='auto')  # Muestra el patrón corrupto.
    plt.title("Patrón Corrupto")  # Título del gráfico.
    plt.subplot(1, 3, 2)  # Segundo subgráfico.
    plt.imshow([hopfield_result], cmap='coolwarm', aspect='auto')  # Muestra el patrón recuperado.
    plt.title("Patrón Recuperado")  # Título del gráfico.
    plt.subplot(1, 3, 3)  # Tercer subgráfico.
    plt.imshow(hebb_weights, cmap='viridis')  # Muestra los pesos aprendidos.
    plt.title("Pesos de Hebb")  # Título del gráfico.
    plt.tight_layout()  # Ajusta el diseño de los gráficos.
    plt.show()  # Muestra los gráficos.

# Punto de entrada principal del programa.
if __name__ == "__main__":
    main()  # Llama a la función principal.