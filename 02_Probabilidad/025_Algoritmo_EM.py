import numpy as np
from scipy.stats import multivariate_normal

class GaussianMixtureEM:
    def __init__(self, n_components, max_iter=100, tol=1e-6):
        """
        Inicializa el modelo de mezcla gaussiana (GMM) con el algoritmo EM.
        
        Parámetros:
        - n_components: Número de componentes (distribuciones gaussianas).
        - max_iter: Número máximo de iteraciones para el algoritmo EM.
        - tol: Tolerancia para el criterio de convergencia.
        """
        self.n_components = n_components
        self.max_iter = max_iter
        self.tol = tol
        self.weights_ = None  # Pesos de cada componente
        self.means_ = None    # Medias de cada componente
        self.covariances_ = None  # Matrices de covarianza de cada componente
    
    def _initialize(self, X):
        """
        Inicializa los parámetros del modelo de forma aleatoria.
        
        Parámetros:
        - X: Datos de entrada (matriz de muestras).
        """
        n_samples, n_features = X.shape
        
        # Inicializar pesos uniformes (iguales para todas las componentes)
        self.weights_ = np.ones(self.n_components) / self.n_components
        
        # Seleccionar puntos aleatorios del conjunto de datos como medias iniciales
        random_idx = np.random.choice(n_samples, self.n_components, replace=False)
        self.means_ = X[random_idx]
        
        # Inicializar las matrices de covarianza como matrices identidad
        self.covariances_ = np.array([np.eye(n_features) for _ in range(self.n_components)])
    
    def _e_step(self, X):
        """
        Paso E (Expectation): Calcula las responsabilidades (probabilidades de pertenencia).
        
        Parámetros:
        - X: Datos de entrada.
        
        Retorna:
        - responsibilities: Matriz de responsabilidades (n_samples x n_components).
        """
        n_samples = X.shape[0]
        responsibilities = np.zeros((n_samples, self.n_components))
        
        for k in range(self.n_components):
            # Crear una distribución normal multivariada para cada componente
            rv = multivariate_normal(
                mean=self.means_[k], 
                cov=self.covariances_[k]
            )
            # Calcular la probabilidad de cada muestra bajo la componente k
            responsibilities[:, k] = self.weights_[k] * rv.pdf(X)
        
        # Normalizar las responsabilidades para que sumen 1 en cada fila
        responsibilities /= responsibilities.sum(axis=1, keepdims=True)
        return responsibilities
    
    def _m_step(self, X, responsibilities):
        """
        Paso M (Maximization): Actualiza los parámetros del modelo.
        
        Parámetros:
        - X: Datos de entrada.
        - responsibilities: Matriz de responsabilidades calculada en el paso E.
        """
        n_samples, n_features = X.shape
        
        # Calcular el número efectivo de muestras asignadas a cada componente
        Nk = responsibilities.sum(axis=0)
        
        # Actualizar los pesos de las componentes
        self.weights_ = Nk / n_samples
        
        # Actualizar las medias de las componentes
        self.means_ = np.zeros((self.n_components, n_features))
        for k in range(self.n_components):
            self.means_[k] = (responsibilities[:, k] @ X) / Nk[k]
        
        # Actualizar las matrices de covarianza
        self.covariances_ = np.zeros((self.n_components, n_features, n_features))
        for k in range(self.n_components):
            diff = X - self.means_[k]
            self.covariances_[k] = (responsibilities[:, k] * diff.T) @ diff / Nk[k]
            # Regularización para evitar matrices singulares
            self.covariances_[k] += 1e-6 * np.eye(n_features)
    
    def fit(self, X):
        """
        Entrena el modelo utilizando el algoritmo EM.
        
        Parámetros:
        - X: Datos de entrada.
        """
        self._initialize(X)  # Inicializar parámetros
        prev_log_likelihood = None  # Log-verosimilitud previa
        
        for iteration in range(self.max_iter):
            # Paso E: Calcular responsabilidades
            responsibilities = self._e_step(X)
            
            # Paso M: Actualizar parámetros
            self._m_step(X, responsibilities)
            
            # Calcular la log-verosimilitud actual
            current_log_likelihood = self._compute_log_likelihood(X)
            
            # Verificar criterio de convergencia
            if prev_log_likelihood is not None:
                if abs(current_log_likelihood - prev_log_likelihood) < self.tol:
                    print(f"Convergencia alcanzada en iteración {iteration}")
                    break
            
            prev_log_likelihood = current_log_likelihood
    
    def _compute_log_likelihood(self, X):
        """
        Calcula la log-verosimilitud del modelo dado el conjunto de datos.
        
        Parámetros:
        - X: Datos de entrada.
        
        Retorna:
        - log_likelihood: Log-verosimilitud total.
        """
        likelihood = np.zeros((X.shape[0], self.n_components))
        for k in range(self.n_components):
            rv = multivariate_normal(
                mean=self.means_[k], 
                cov=self.covariances_[k]
            )
            likelihood[:, k] = self.weights_[k] * rv.pdf(X)
        return np.log(likelihood.sum(axis=1)).sum()
    
    def predict(self, X):
        """
        Asigna cada muestra al componente más probable.
        
        Parámetros:
        - X: Datos de entrada.
        
        Retorna:
        - labels: Índices de los componentes asignados a cada muestra.
        """
        responsibilities = self._e_step(X)
        return np.argmax(responsibilities, axis=1)

# Ejemplo con datos sintéticos
np.random.seed(42)

# Generar datos de tres distribuciones normales
n_samples = 300
X1 = np.random.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n_samples//3)
X2 = np.random.multivariate_normal([5, 5], [[1, -0.7], [-0.7, 1]], n_samples//3)
X3 = np.random.multivariate_normal([-5, 5], [[0.5, 0], [0, 0.5]], n_samples//3)
X = np.vstack([X1, X2, X3])

# Aplicar EM
gmm = GaussianMixtureEM(n_components=3)
gmm.fit(X)

# Resultados
print("Pesos estimados:", gmm.weights_)
print("Medias estimadas:\n", gmm.means_)
print("Covarianzas estimadas:\n", gmm.covariances_)