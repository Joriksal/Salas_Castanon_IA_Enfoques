import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt  # Importación para visualización

class HMM:
    def __init__(self, n_states):
        """
        Inicializa un Modelo de Markov Oculto (HMM).
        :param n_states: Número de estados ocultos.
        """
        self.n_states = n_states
        self.transition_matrix = None  # Matriz de transición entre estados
        self.emission_means = None  # Medias de las distribuciones de emisión
        self.emission_stds = None  # Desviaciones estándar de las distribuciones de emisión
        self.start_prob = None  # Probabilidades iniciales de los estados
    
    def initialize_params(self, X):
        """
        Inicializa los parámetros del HMM usando K-Means para las emisiones.
        :param X: Secuencia de observaciones.
        """
        n_samples = X.shape[0]
        
        # Inicialización uniforme de la matriz de transición y probabilidades iniciales
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        self.start_prob = np.ones(self.n_states) / self.n_states
        
        # Uso de K-Means para inicializar las medias de las emisiones
        from sklearn.cluster import KMeans
        kmeans = KMeans(n_clusters=self.n_states).fit(X.reshape(-1, 1))
        self.emission_means = np.sort(kmeans.cluster_centers_.flatten())
        self.emission_stds = np.array([X.std()] * self.n_states)  # Inicialización uniforme de desviaciones estándar
    
    def forward(self, X):
        """
        Implementa el algoritmo hacia adelante para calcular las probabilidades.
        :param X: Secuencia de observaciones.
        :return: Matriz alpha con probabilidades hacia adelante.
        """
        n_samples = len(X)
        alpha = np.zeros((n_samples, self.n_states))
        
        # Paso inicial: calcular alpha para el primer tiempo
        alpha[0] = self.start_prob * norm.pdf(X[0], self.emission_means, self.emission_stds)
        
        # Pasos recursivos: calcular alpha para los tiempos posteriores
        for t in range(1, n_samples):
            for j in range(self.n_states):
                alpha[t, j] = norm.pdf(X[t], self.emission_means[j], self.emission_stds[j]) * \
                              np.sum(alpha[t-1] * self.transition_matrix[:, j])
        
        return alpha
    
    def backward(self, X):
        """
        Implementa el algoritmo hacia atrás para calcular las probabilidades.
        :param X: Secuencia de observaciones.
        :return: Matriz beta con probabilidades hacia atrás.
        """
        n_samples = len(X)
        beta = np.zeros((n_samples, self.n_states))
        
        # Paso final: inicializar beta en el último tiempo
        beta[-1] = 1.0
        
        # Pasos recursivos: calcular beta hacia atrás
        for t in range(n_samples-2, -1, -1):
            for i in range(self.n_states):
                beta[t, i] = np.sum(
                    self.transition_matrix[i, :] * 
                    norm.pdf(X[t+1], self.emission_means, self.emission_stds) * 
                    beta[t+1, :]
                )
        
        return beta
    
    def baum_welch(self, X, n_iter=100):
        """
        Implementa el algoritmo Baum-Welch (EM) para entrenar el HMM.
        :param X: Secuencia de observaciones.
        :param n_iter: Número de iteraciones.
        """
        n_samples = len(X)
        
        for _ in range(n_iter):
            # Paso E: calcular alpha y beta
            alpha = self.forward(X)
            beta = self.backward(X)
            
            # Calcular probabilidades suavizadas (gamma)
            gamma = alpha * beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)
            
            # Calcular probabilidades de transición (xi)
            xi = np.zeros((n_samples-1, self.n_states, self.n_states))
            for t in range(n_samples-1):
                xi[t, :, :] = alpha[t, :, None] * self.transition_matrix * \
                              norm.pdf(X[t+1], self.emission_means, self.emission_stds) * \
                              beta[t+1, None, :]
                xi[t, :, :] /= np.sum(xi[t, :, :])
            
            # Paso M: actualizar parámetros
            self.start_prob = gamma[0]
            self.transition_matrix = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True).T
            
            for k in range(self.n_states):
                self.emission_means[k] = np.sum(gamma[:, k] * X) / np.sum(gamma[:, k])
                self.emission_stds[k] = np.sqrt(
                    np.sum(gamma[:, k] * (X - self.emission_means[k])**2) / np.sum(gamma[:, k])
                )
    
    def viterbi(self, X):
        """
        Implementa el algoritmo de Viterbi para encontrar la secuencia más probable de estados ocultos.
        :param X: Secuencia de observaciones.
        :return: Secuencia de estados más probable.
        """
        n_samples = len(X)
        delta = np.zeros((n_samples, self.n_states))  # Probabilidades máximas
        psi = np.zeros((n_samples, self.n_states), dtype=int)  # Índices de estados previos
        
        # Inicialización
        delta[0] = self.start_prob * norm.pdf(X[0], self.emission_means, self.emission_stds)
        
        # Recursión: calcular delta y psi
        for t in range(1, n_samples):
            for j in range(self.n_states):
                trans_probs = delta[t-1] * self.transition_matrix[:, j]
                psi[t, j] = np.argmax(trans_probs)
                delta[t, j] = trans_probs[psi[t, j]] * \
                              norm.pdf(X[t], self.emission_means[j], self.emission_stds[j])
        
        # Backtracking: reconstruir la secuencia de estados
        path = np.zeros(n_samples, dtype=int)
        path[-1] = np.argmax(delta[-1])
        
        for t in range(n_samples-2, -1, -1):
            path[t] = psi[t+1, path[t+1]]
        
        return path

# Ejemplo con datos simulados
np.random.seed(42)
n_samples = 200
tendencia = np.linspace(0, 1, n_samples)  # Tendencia lineal
X = tendencia + np.random.normal(0, 0.2, n_samples)  # Observaciones con ruido

# Crear y entrenar el HMM
hmm = HMM(n_states=3)  # Modelo con 3 estados ocultos
hmm.initialize_params(X)  # Inicializar parámetros
hmm.baum_welch(X, n_iter=50)  # Entrenar el modelo

# Decodificar la secuencia más probable de estados ocultos
estados = hmm.viterbi(X)

# Visualización de los resultados
plt.figure(figsize=(12, 6))
plt.plot(X, 'b-', label='Observaciones')  # Observaciones
plt.plot(estados, 'r--', label='Estados ocultos')  # Estados ocultos
plt.title('Modelo de Markov Oculto: Predicción de Estados')
plt.xlabel('Tiempo')
plt.ylabel('Valor')
plt.legend()
plt.show()