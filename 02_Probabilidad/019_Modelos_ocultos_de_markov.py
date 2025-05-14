import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos numéricos.
                    # Es útil para:
                    # - Representar y manipular matrices como las de transición (A) y emisión (B).
                    # - Realizar cálculos eficientes con operaciones vectorizadas.
                    # - Calcular probabilidades hacia adelante y hacia atrás en el algoritmo.

# --- Definir el HMM ---
# Estados ocultos del modelo (fonemas + silencio)
states = ["silencioso", "A", "B", "C"]  

# Observaciones posibles (intensidad de audio)
observations = ["bajo", "medio", "alto"]  

# Probabilidades iniciales (π): Probabilidad de comenzar en cada estado
pi = np.array([0.6, 0.2, 0.1, 0.1])

# Matriz de transición (A): Probabilidad de pasar de un estado a otro
A = np.array([
    [0.7, 0.2, 0.1, 0.0],  # Desde "silencioso"
    [0.0, 0.5, 0.3, 0.2],  # Desde "A"
    [0.0, 0.3, 0.4, 0.3],  # Desde "B"
    [0.1, 0.1, 0.2, 0.6]   # Desde "C"
])

# Matriz de emisión (B): Probabilidad de observar una intensidad dada un estado
B = np.array([
    [0.8, 0.1, 0.1],  # Silencioso → bajo
    [0.1, 0.7, 0.2],  # A → medio
    [0.2, 0.6, 0.2],  # B → medio/alto
    [0.1, 0.3, 0.6]   # C → alto
])

# --- Algoritmo de Viterbi ---
def viterbi(obs, pi, A, B):
    """
    Implementación del algoritmo de Viterbi para encontrar la secuencia más probable
    de estados ocultos en un HMM dado un conjunto de observaciones.

    Parámetros:
    - obs: Secuencia de observaciones (índices de las observaciones)
    - pi: Vector de probabilidades iniciales
    - A: Matriz de transición
    - B: Matriz de emisión

    Retorna:
    - path: Secuencia más probable de estados ocultos
    """
    T = len(obs)  # Longitud de la secuencia de observaciones
    N = len(pi)   # Número de estados en el HMM

    # Matriz delta: Almacena las probabilidades máximas hasta cada estado en cada tiempo
    delta = np.zeros((T, N))

    # Matriz psi: Almacena los índices de los estados previos que maximizan delta
    psi = np.zeros((T, N), dtype=int)

    # --- Inicialización ---
    # Calcula las probabilidades iniciales para el primer tiempo
    delta[0] = pi * B[:, obs[0]]

    # --- Recursión ---
    # Itera sobre el tiempo t y los estados j para calcular delta y psi
    for t in range(1, T):
        for j in range(N):
            # Calcula la probabilidad de llegar al estado j desde cualquier estado previo
            prob = delta[t-1] * A[:, j]
            # Encuentra el estado previo que maximiza la probabilidad
            psi[t, j] = np.argmax(prob)
            # Almacena la probabilidad máxima en delta
            delta[t, j] = np.max(prob) * B[j, obs[t]]

    # --- Backtracking ---
    # Reconstruye la secuencia más probable de estados ocultos
    path = np.zeros(T, dtype=int)
    # Encuentra el estado más probable en el último tiempo
    path[-1] = np.argmax(delta[-1])
    # Retrocede en el tiempo para determinar los estados previos
    for t in range(T-2, -1, -1):
        path[t] = psi[t+1, path[t+1]]

    return path

# --- Ejemplo: Decodificar una palabra ---
# Secuencia de observaciones: ["medio", "alto", "medio", "bajo"]
# Representada como índices: 1=medio, 2=alto, 0=bajo
obs_seq = [1, 2, 1, 0]

# Decodifica la secuencia más probable de estados ocultos
estados_decodificados = viterbi(obs_seq, pi, A, B)

# Imprime la secuencia de estados más probable
print("Secuencia de fonemas más probable:")
for t, estado in enumerate(estados_decodificados):
    print(f"Tiempo {t+1}: {states[estado]}")