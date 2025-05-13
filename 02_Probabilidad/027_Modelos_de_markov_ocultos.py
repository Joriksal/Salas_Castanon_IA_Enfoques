# Importación de librerías necesarias
import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos multidimensionales
from scipy.stats import norm  # Módulo de SciPy para trabajar con distribuciones estadísticas, como la normal
import matplotlib.pyplot as plt  # Librería para crear gráficos y visualizaciones

# Definición de la clase HMM (Hidden Markov Model)
class HMM:
    def __init__(self, n_states):
        """
        Inicializa un Modelo de Markov Oculto (HMM).
        :param n_states: Número de estados ocultos.
        """
        self.n_states = n_states  # Número de estados ocultos en el modelo
        self.transition_matrix = None  # Matriz de transición entre estados (probabilidades de pasar de un estado a otro)
        self.emission_means = None  # Medias de las distribuciones de emisión (valores esperados de las observaciones)
        self.emission_stds = None  # Desviaciones estándar de las distribuciones de emisión
        self.start_prob = None  # Probabilidades iniciales de los estados (probabilidad de comenzar en cada estado)

    def initialize_params(self, X):
        """
        Inicializa los parámetros del HMM usando K-Means para las emisiones.
        :param X: Secuencia de observaciones.
        """
        n_samples = X.shape[0]  # Número de observaciones en la secuencia

        # Inicialización uniforme de la matriz de transición y probabilidades iniciales
        self.transition_matrix = np.ones((self.n_states, self.n_states)) / self.n_states
        # np.ones crea una matriz llena de unos; aquí se divide por el número de estados para que las probabilidades sean uniformes
        self.start_prob = np.ones(self.n_states) / self.n_states  # Probabilidades iniciales uniformes

        # Uso de K-Means para inicializar las medias de las emisiones
        from sklearn.cluster import KMeans  # Importación del algoritmo K-Means para agrupamiento
        kmeans = KMeans(n_clusters=self.n_states).fit(X.reshape(-1, 1))  # Ajusta K-Means a los datos
        self.emission_means = np.sort(kmeans.cluster_centers_.flatten())  # Ordena las medias de los clústeres
        self.emission_stds = np.array([X.std()] * self.n_states)  # Inicializa las desviaciones estándar como el desvío global

    def forward(self, X):
        """
        Implementa el algoritmo hacia adelante para calcular las probabilidades.
        :param X: Secuencia de observaciones.
        :return: Matriz alpha con probabilidades hacia adelante.
        """
        n_samples = len(X)  # Número de observaciones
        alpha = np.zeros((n_samples, self.n_states))  # Matriz para almacenar las probabilidades hacia adelante

        # Paso inicial: calcular alpha para el primer tiempo
        alpha[0] = self.start_prob * norm.pdf(X[0], self.emission_means, self.emission_stds)
        # norm.pdf calcula la densidad de probabilidad de la distribución normal para cada estado

        # Pasos recursivos: calcular alpha para los tiempos posteriores
        for t in range(1, n_samples):  # Itera sobre cada tiempo t
            for j in range(self.n_states):  # Itera sobre cada estado j
                alpha[t, j] = norm.pdf(X[t], self.emission_means[j], self.emission_stds[j]) * \
                              np.sum(alpha[t-1] * self.transition_matrix[:, j])
                # np.sum suma las probabilidades ponderadas por la matriz de transición

        return alpha  # Devuelve la matriz alpha

    def backward(self, X):
        """
        Implementa el algoritmo hacia atrás para calcular las probabilidades.
        :param X: Secuencia de observaciones.
        :return: Matriz beta con probabilidades hacia atrás.
        """
        n_samples = len(X)  # Número de observaciones
        beta = np.zeros((n_samples, self.n_states))  # Matriz para almacenar las probabilidades hacia atrás

        # Paso final: inicializar beta en el último tiempo
        beta[-1] = 1.0  # Última fila de beta se inicializa en 1 (probabilidad completa)

        # Pasos recursivos: calcular beta hacia atrás
        for t in range(n_samples-2, -1, -1):  # Itera hacia atrás desde el penúltimo tiempo
            for i in range(self.n_states):  # Itera sobre cada estado i
                beta[t, i] = np.sum(
                    self.transition_matrix[i, :] * 
                    norm.pdf(X[t+1], self.emission_means, self.emission_stds) * 
                    beta[t+1, :]
                )
                # Calcula la probabilidad acumulada hacia atrás usando la matriz de transición y beta del siguiente tiempo

        return beta  # Devuelve la matriz beta

    def baum_welch(self, X, n_iter=100):
        """
        Implementa el algoritmo Baum-Welch (EM) para entrenar el HMM.
        :param X: Secuencia de observaciones.
        :param n_iter: Número de iteraciones.
        """
        n_samples = len(X)  # Número de observaciones

        for _ in range(n_iter):  # Itera el número de veces especificado
            # Paso E: calcular alpha y beta
            alpha = self.forward(X)  # Probabilidades hacia adelante
            beta = self.backward(X)  # Probabilidades hacia atrás

            # Calcular probabilidades suavizadas (gamma)
            gamma = alpha * beta  # Producto de alpha y beta
            gamma /= np.sum(gamma, axis=1, keepdims=True)  # Normalización para que las probabilidades sumen 1

            # Calcular probabilidades de transición (xi)
            xi = np.zeros((n_samples-1, self.n_states, self.n_states))  # Tensor para almacenar xi
            for t in range(n_samples-1):  # Itera sobre cada tiempo
                xi[t, :, :] = alpha[t, :, None] * self.transition_matrix * \
                              norm.pdf(X[t+1], self.emission_means, self.emission_stds) * \
                              beta[t+1, None, :]
                xi[t, :, :] /= np.sum(xi[t, :, :])  # Normalización

            # Paso M: actualizar parámetros
            self.start_prob = gamma[0]  # Actualiza las probabilidades iniciales
            self.transition_matrix = np.sum(xi, axis=0) / np.sum(gamma[:-1], axis=0, keepdims=True).T
            # Actualiza la matriz de transición

            for k in range(self.n_states):  # Actualiza las medias y desviaciones estándar de las emisiones
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
        n_samples = len(X)  # Número de observaciones
        delta = np.zeros((n_samples, self.n_states))  # Probabilidades máximas
        psi = np.zeros((n_samples, self.n_states), dtype=int)  # Índices de estados previos

        # Inicialización
        delta[0] = self.start_prob * norm.pdf(X[0], self.emission_means, self.emission_stds)

        # Recursión: calcular delta y psi
        for t in range(1, n_samples):  # Itera sobre cada tiempo
            for j in range(self.n_states):  # Itera sobre cada estado
                trans_probs = delta[t-1] * self.transition_matrix[:, j]
                psi[t, j] = np.argmax(trans_probs)  # Encuentra el índice del estado previo más probable
                delta[t, j] = trans_probs[psi[t, j]] * \
                              norm.pdf(X[t], self.emission_means[j], self.emission_stds[j])

        # Backtracking: reconstruir la secuencia de estados
        path = np.zeros(n_samples, dtype=int)  # Secuencia de estados más probable
        path[-1] = np.argmax(delta[-1])  # Último estado más probable

        for t in range(n_samples-2, -1, -1):  # Itera hacia atrás para reconstruir la secuencia
            path[t] = psi[t+1, path[t+1]]

        return path  # Devuelve la secuencia de estados más probable

# Ejemplo con datos simulados
np.random.seed(42)  # Fija la semilla para reproducibilidad
n_samples = 200  # Número de observaciones
tendencia = np.linspace(0, 1, n_samples)  # Genera una tendencia lineal
X = tendencia + np.random.normal(0, 0.2, n_samples)  # Agrega ruido gaussiano a la tendencia

# Crear y entrenar el HMM
hmm = HMM(n_states=3)  # Modelo con 3 estados ocultos
hmm.initialize_params(X)  # Inicializar parámetros
hmm.baum_welch(X, n_iter=50)  # Entrenar el modelo con 50 iteraciones

# Decodificar la secuencia más probable de estados ocultos
estados = hmm.viterbi(X)

# Visualización de los resultados
plt.figure(figsize=(12, 6))  # Configura el tamaño del gráfico
plt.plot(X, 'b-', label='Observaciones')  # Dibuja las observaciones
plt.plot(estados, 'r--', label='Estados ocultos')  # Dibuja los estados ocultos
plt.title('Modelo de Markov Oculto: Predicción de Estados')  # Título del gráfico
plt.xlabel('Tiempo')  # Etiqueta del eje X
plt.ylabel('Valor')  # Etiqueta del eje Y
plt.legend()  # Muestra la leyenda
plt.show()  # Muestra el gráfico