import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

# --- 1. Red de Hamming (Clasificación) ---
class HammingNetwork:
    def __init__(self, num_patterns, pattern_size):
        # Inicializa los pesos como una matriz de ceros
        self.weights = np.zeros((num_patterns, pattern_size))
    
    def train(self, patterns):
        # Entrena la red copiando los patrones directamente como pesos
        self.weights = patterns.copy()
    
    def predict(self, input_pattern):
        # Calcula la similitud entre el patrón de entrada y los patrones entrenados
        similarities = np.dot(self.weights, input_pattern)
        return np.argmax(similarities)  # Devuelve el índice del patrón más similar

# --- 2. Red de Hopfield (Memoria Asociativa) ---
class HopfieldNetwork:
    def __init__(self, size):
        # Inicializa los pesos como una matriz de ceros
        self.weights = np.zeros((size, size))
    
    def train(self, patterns):
        # Entrena la red utilizando la regla de aprendizaje de Hebb
        for p in patterns:
            self.weights += np.outer(p, p)  # Calcula el producto externo de cada patrón
        np.fill_diagonal(self.weights, 0)  # Elimina autoconexiones (diagonal en ceros)
    
    def predict(self, input_pattern, max_iter=100):
        # Recupera un patrón a partir de una entrada corrupta
        pattern = input_pattern.copy()
        for _ in range(max_iter):
            # Actualiza el patrón utilizando la regla de activación
            new_pattern = np.sign(np.dot(self.weights, pattern))
            if np.array_equal(new_pattern, pattern):  # Si converge, detiene la iteración
                break
            pattern = new_pattern
        return pattern

# --- 3. Regla de Hebb (Aprendizaje No Supervisado) ---
def hebb_rule(input_data, output_data, lr=0.1):
    # Inicializa los pesos como una matriz de ceros
    weights = np.zeros((output_data.shape[1], input_data.shape[1]))
    for x, y in zip(input_data, output_data):
        # Actualiza los pesos utilizando la regla de Hebb
        weights += lr * np.outer(y, x)
    return weights

# --- 4. Máquina de Boltzmann (Generativa) ---
class BoltzmannMachine(nn.Module):
    def __init__(self, num_visible, num_hidden):
        super().__init__()
        # Inicializa los pesos y sesgos de la máquina de Boltzmann
        self.W = nn.Parameter(torch.randn(num_hidden, num_visible) * 0.1)  # Pesos
        self.b = nn.Parameter(torch.zeros(num_hidden))  # Sesgos
    
    def forward(self, v):
        # Calcula las probabilidades de activación de las neuronas ocultas
        h_prob = torch.sigmoid(torch.matmul(self.W, v.t()) + self.b)
        return h_prob

# --- Datos de Ejemplo ---
# Hamming y Hopfield: Patrones binarios
patterns = np.array([
    [1, 1, -1, -1],  # Patrón A
    [-1, -1, 1, 1]   # Patrón B
])
test_pattern = np.array([1, -1, -1, -1])  # Patrón corrupto (debería recuperar A)

# Hebb: Entradas/Salidas (ejemplo simplificado)
X_hebb = np.array([[1, -1], [-1, 1]])  # Entradas
Y_hebb = np.array([[1, 0], [0, 1]])  # Salidas deseadas

# Boltzmann: Datos aleatorios (simulados)
data_boltzmann = torch.tensor([[1, 0, 1], [0, 1, 0]], dtype=torch.float)

# --- Ejecución y Visualización ---
def main():
    # 1. Hamming
    hn_hamming = HammingNetwork(num_patterns=2, pattern_size=4)
    hn_hamming.train(patterns)  # Entrena con los patrones binarios
    hamming_result = hn_hamming.predict(test_pattern)  # Clasifica el patrón corrupto
    print(f"Hamming: Patrón clasificado como {hamming_result}")

    # 2. Hopfield
    hn_hopfield = HopfieldNetwork(size=4)
    hn_hopfield.train(patterns)  # Entrena con los patrones binarios
    hopfield_result = hn_hopfield.predict(test_pattern)  # Recupera el patrón original
    print(f"Hopfield: Patrón recuperado:\n{hopfield_result}")

    # 3. Hebb
    hebb_weights = hebb_rule(X_hebb, Y_hebb)  # Calcula los pesos con la regla de Hebb
    print(f"Hebb: Pesos aprendidos:\n{hebb_weights}")

    # 4. Boltzmann (ejemplo simplificado)
    bm = BoltzmannMachine(num_visible=3, num_hidden=2)
    with torch.no_grad():
        sample = torch.tensor([1, 0, 1], dtype=torch.float)  # Entrada de ejemplo
        boltzmann_result = bm(sample).numpy()  # Calcula las probabilidades de las neuronas ocultas
    print(f"Boltzmann: Probabilidad de neuronas ocultas:\n{boltzmann_result}")

    # Visualización (Hopfield)
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 3, 1)
    plt.imshow([test_pattern], cmap='coolwarm', aspect='auto')
    plt.title("Patrón Corrupto")
    plt.subplot(1, 3, 2)
    plt.imshow([hopfield_result], cmap='coolwarm', aspect='auto')
    plt.title("Patrón Recuperado")
    plt.subplot(1, 3, 3)
    plt.imshow(hebb_weights, cmap='viridis')
    plt.title("Pesos de Hebb")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()