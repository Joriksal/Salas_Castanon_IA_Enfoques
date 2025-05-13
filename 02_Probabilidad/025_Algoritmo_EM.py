# Importamos las librerías necesarias
import numpy as np  # Librería para trabajar con arreglos y operaciones matemáticas de alto rendimiento
from scipy.stats import multivariate_normal  # Para manejar distribuciones normales multivariadas

# Definimos la clase GaussianMixtureEM que implementa el algoritmo EM para modelos de mezcla gaussiana
class GaussianMixtureEM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        """
        Constructor de la clase. Inicializa los parámetros del modelo.
        
        Parámetros:
        - n_components: Número de componentes gaussianas en la mezcla.
        - max_iter: Número máximo de iteraciones para el algoritmo EM.
        - tol: Tolerancia para determinar la convergencia del algoritmo.
        """
        self.n_components = n_components  # Número de componentes gaussianas
        self.max_iter = max_iter  # Límite de iteraciones para evitar bucles infinitos
        self.tol = tol  # Tolerancia para el criterio de convergencia
        self.weights_ = None  # Pesos iniciales de las componentes (probabilidades a priori)
        self.means_ = None  # Medias iniciales de las componentes gaussianas
        self.covariances_ = None  # Matrices de covarianza iniciales de las componentes

    def _initialize(self, X):
        """
        Inicializa los parámetros del modelo de forma aleatoria.
        
        Parámetros:
        - X: Matriz de datos de entrada (n_samples x n_features).
        """
        n_samples, n_features = X.shape  # Obtiene el número de muestras y características
        
        # Inicializamos los pesos de las componentes de forma uniforme (iguales para todas)
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Seleccionamos aleatoriamente puntos del conjunto de datos como medias iniciales
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_idx]
        
        # Inicializamos las matrices de covarianza como matrices identidad
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])

    def _e_step(self, X):
        """
        Paso E (Expectation): Calcula las responsabilidades (probabilidades de pertenencia).
        
        Parámetros:
        - X: Matriz de datos de entrada.
        
        Retorna:
        - responsibilities: Matriz de responsabilidades (n_samples x n_components).
        """
        n_samples = X.shape[0]  # Número de muestras
        responsibilities = np.zeros((n_samples, self.n_components))  # Inicializamos la matriz de responsabilidades
        
        for k in range(self.n_components):
            # Creamos una distribución normal multivariada para cada componente
            rv = multivariate_normal(
                mean=self.means_[k],  # Media de la componente k
                cov=self.covariances_[k]  # Matriz de covarianza de la componente k
            )
            # Calculamos la probabilidad de cada muestra bajo la componente k
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
        
        # Normalizamos las responsabilidades para que sumen 1 en cada fila
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities

    def _m_step(self, X, responsibilities):
        """
        Paso M (Maximization): Actualiza los parámetros del modelo.
        
        Parámetros:
        - X: Matriz de datos de entrada.
        - responsibilities: Matriz de responsabilidades calculada en el paso E.
        """
        n_samples, n_features = X.shape  # Dimensiones de los datos
        
        # Calculamos el número efectivo de muestras asignadas a cada componente
        Nk = responsibilities.sum(axis=0)  # Suma de responsabilidades por componente
        
        # Actualizamos los pesos de las componentes (probabilidades a priori)
        self.weights_ = Nk / n_samples
        
        # Actualizamos las medias de las componentes
        self.means_ = np.zeros((self.n_components, n_features))  # Inicializamos las medias
        for k in range(self.n_components):
            # Calculamos la nueva media ponderada por las responsabilidades
            self.means_[k] = (responsibilities[:, k] @ X) / Nk[k]
        
        # Actualizamos las matrices de covarianza
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))  # Inicializamos las covarianzas
        for k in range(self.n_components):
            diff = X - self.means_[k]  # Diferencia entre los datos y la media
            # Calculamos la nueva matriz de covarianza ponderada por las responsabilidades
            self.covariances_[k] = (responsibilities[:, k] * diff.T) @ diff / Nk[k]
            # Añadimos un término de regularización para evitar matrices singulares
            self.covariances_[k] += 1e-6 * np.eye(n_features)

    def fit(self, X):
        """
        Entrena el modelo utilizando el algoritmo EM.
        
        Parámetros:
        - X: Matriz de datos de entrada.
        """
        self._initialize(X)  # Inicializamos los parámetros
        prev_log_likelihood = None  # Log-verosimilitud previa para verificar convergencia
        
        for iteration in range(self.max_iter):
            # Paso E: Calculamos las responsabilidades
            responsibilities = self._e_step(X)
            
            # Paso M: Actualizamos los parámetros
            self._m_step(X, responsibilities)
            
            # Calculamos la log-verosimilitud actual
            current_log_likelihood = self._compute_log_likelihood(X)
            
            # Verificamos el criterio de convergencia
            if prev_log_likelihood is not None:
                if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"Convergencia alcanzada en iteración {iteration}")
                    break
            
            prev_log_likelihood = current_log_likelihood

    def _compute_log_likelihood(self, X):
        """
        Calcula la log-verosimilitud del modelo dado el conjunto de datos.
        
        Parámetros:
        - X: Matriz de datos de entrada.
        
        Retorna:
        - log_likelihood: Log-verosimilitud total.
        """
        likelihood = np.zeros((X.shape[0], self.n_components))  # Inicializamos la matriz de verosimilitudes
        for k in range(self.n_components):
            # Creamos una distribución normal multivariada para cada componente
            rv = multivariate_normal(
                mean=self.means_[k], 
                cov=self.covariances_[k]
            )
            # Calculamos la probabilidad de cada muestra bajo la componente k
            likelihood[:, k] = self.weights_[k] * rv.pdf(X)
        # Sumamos las probabilidades y calculamos el logaritmo
        return np.log(likelihood.sum(axis=1)).sum()

    def predict(self, X):
        """
        Asigna cada muestra al componente más probable.
        
        Parámetros:
        - X: Matriz de datos de entrada.
        
        Retorna:
        - labels: Índices de los componentes asignados a cada muestra.
        """
        responsibilities = self._e_step(X)  # Calculamos las responsabilidades
        return np.argmax(responsibilities, axis=1)  # Asignamos la muestra al componente con mayor responsabilidad

# Ejemplo con datos sintéticos
np.random.seed(42)  # Fijamos la semilla para reproducibilidad

# Generamos datos de tres distribuciones normales
n_samples = 300
X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//3)
X2 = np.random.multivariate_normal([5, 5], [[1, -0.7], [-0.7, 1]], n_samples//3)
X3 = np.random.multivariate_normal([-5, 5], [[0.5, 0], [0, 0.5]], n_samples//3)
X = np.vstack([X1, X2, X3])  # Combinamos los datos en una sola matriz

# Aplicamos el algoritmo EM
gmm = GaussianMixtureEM(n_components=3)  # Creamos el modelo con 3 componentes
gmm.fit(X)  # Entrenamos el modelo con los datos

# Mostramos los resultados
print("Pesos estimados:", gmm.weights_)  # Pesos finales de las componentes
print("Medias estimadas:\n", gmm.means_)  # Medias finales de las componentes
print("Covarianzas estimadas:\n", gmm.covariances_)  # Covarianzas finales de las componentes