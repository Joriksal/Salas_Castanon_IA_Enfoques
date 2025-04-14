import numpy as np

# --- Función para el paso hacia adelante ---
def forward_algorithm(obs, pi, A, B):
    """
    Paso hacia adelante: Calcula la probabilidad de la secuencia de observaciones.
    Args:
        obs: Secuencia de observaciones (lista de enteros).
        pi: Distribución inicial de los estados ocultos.
        A: Matriz de transición de estados.
        B: Matriz de emisión (probabilidad de observaciones dado un estado).
    Returns:
        alpha: Matriz con las probabilidades hacia adelante.
        P_obs: Probabilidad total de la secuencia de observaciones.
    """
    T = len(obs)  # Longitud de la secuencia de observaciones
    N = len(pi)   # Número de estados ocultos
    alpha = np.zeros((T, N))  # Inicialización de la matriz alpha
    
    # Inicialización (t = 0)
    alpha[0] = pi * B[:, obs[0]]  # P(estado inicial y primera observación)
    
    # Inducción (t = 1, ..., T-1)
    for t in range(1, T):
        for j in range(N):
            # Suma de probabilidades ponderadas por la matriz de transición y emisión
            alpha[t, j] = np.sum(alpha[t-1] * A[:, j]) * B[j, obs[t]]
    
    # Probabilidad total de la secuencia de observaciones
    P_obs = np.sum(alpha[-1])
    return alpha, P_obs

# --- Función para el paso hacia atrás ---
def backward_algorithm(obs, A, B):
    """
    Paso hacia atrás: Calcula probabilidades hacia atrás.
    Args:
        obs: Secuencia de observaciones (lista de enteros).
        A: Matriz de transición de estados.
        B: Matriz de emisión (probabilidad de observaciones dado un estado).
    Returns:
        beta: Matriz con las probabilidades hacia atrás.
    """
    T = len(obs)  # Longitud de la secuencia de observaciones
    N = A.shape[0]  # Número de estados ocultos
    beta = np.zeros((T, N))  # Inicialización de la matriz beta
    
    # Inicialización (t = T-1)
    beta[-1] = 1  # Última fila de beta es 1 (probabilidad completa)
    
    # Inducción hacia atrás (t = T-2, ..., 0)
    for t in range(T-2, -1, -1):
        for i in range(N):
            # Suma de probabilidades ponderadas por la matriz de transición y emisión
            beta[t, i] = np.sum(A[i, :] * B[:, obs[t+1]] * beta[t+1, :])
    
    return beta

# --- Algoritmo completo hacia delante-atrás ---
def forward_backward(obs, pi, A, B):
    """
    Algoritmo hacia delante-atrás: Calcula las probabilidades suavizadas de los estados ocultos.
    Args:
        obs: Secuencia de observaciones (lista de enteros).
        pi: Distribución inicial de los estados ocultos.
        A: Matriz de transición de estados.
        B: Matriz de emisión (probabilidad de observaciones dado un estado).
    Returns:
        gamma: Matriz con las probabilidades suavizadas de los estados ocultos.
    """
    # Paso hacia adelante
    alpha, P_obs = forward_algorithm(obs, pi, A, B)
    
    # Paso hacia atrás
    beta = backward_algorithm(obs, A, B)
    
    # Cálculo de las probabilidades suavizadas (gamma)
    gamma = alpha * beta / P_obs  # Normalización por la probabilidad total
    return gamma

# --- Parámetros del HMM ---
# Estados ocultos: 0=caminar, 1=correr, 2=detenido
pi = np.array([0.3, 0.4, 0.3])  # Distribución inicial de los estados

# Matriz de transición (A[i,j] = P(estado_t+1 = j | estado_t = i))
A = np.array([
    [0.7, 0.2, 0.1],  # Desde caminar
    [0.1, 0.6, 0.3],  # Desde correr
    [0.2, 0.3, 0.5]   # Desde detenido
])

# Matriz de emisión (B[i,o] = P(obs = o | estado = i))
# Observaciones: 0=bajo, 1=medio, 2=alto
B = np.array([
    [0.6, 0.3, 0.1],  # Caminar emite bajo/medio/alto
    [0.1, 0.4, 0.5],  # Correr emite más "alto"
    [0.7, 0.2, 0.1]   # Detenido emite más "bajo"
])

# --- Secuencia de observaciones ---
# Ejemplo: [bajo, medio, alto, bajo] ~ ¿Qué actividad hizo la persona?
obs_seq = [0, 1, 2, 0]

# --- Ejecución del algoritmo ---
gamma = forward_backward(obs_seq, pi, A, B)

# --- Resultados ---
state_names = ["Caminar", "Correr", "Detenido"]  # Nombres de los estados ocultos
print("Probabilidades de estados ocultos en cada paso:")
for t in range(len(obs_seq)):
    probs = gamma[t] / np.sum(gamma[t])  # Normalización para asegurar que sumen 1
    print(f"Tiempo {t+1}:")
    for i, state in enumerate(state_names):
        print(f"  P({state}) = {probs[i]:.4f}")
    print()