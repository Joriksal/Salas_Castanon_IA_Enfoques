import numpy as np
from collections import defaultdict
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class NaiveBayesClassifier:
    def __init__(self, alpha=1.0):
        """
        Inicializa el clasificador Naïve Bayes.
        alpha: Parámetro de suavizado de Laplace (evita probabilidades de 0).
        """
        self.alpha = alpha  # Suavizado de Laplace
        self.class_probs = {}  # Probabilidades a priori de las clases P(y)
        self.feature_probs = {}  # Probabilidades condicionales P(x_i|y)
        self.classes = None  # Lista de clases únicas en los datos

    def fit(self, X, y):
        """
        Entrena el modelo Naïve Bayes.
        X: Matriz de características (n_samples, n_features).
        y: Vector de etiquetas/clases (n_samples).
        """
        self.classes = np.unique(y)  # Identificar las clases únicas
        n_samples, n_features = X.shape  # Número de muestras y características
        
        # 1. Calcular probabilidades a priori P(y)
        for cls in self.classes:
            self.class_probs[cls] = np.sum(y == cls) / n_samples  # Frecuencia relativa de cada clase
        
        # 2. Calcular probabilidades condicionales P(x_i|y)
        self.feature_probs = {cls: defaultdict(list) for cls in self.classes}  # Inicializar estructura para almacenar probabilidades
        
        for cls in self.classes:
            cls_samples = X[y == cls]  # Filtrar muestras de la clase actual
            for feature_idx in range(n_features):
                feature_values = cls_samples[:, feature_idx]  # Valores de la característica actual
                # Verificar si la característica es discreta o continua
                if self._is_discrete(feature_values):
                    # Caso discreto: calcular frecuencias relativas con suavizado de Laplace
                    counts = np.bincount(feature_values.astype(int))  # Conteo de valores únicos
                    probs = (counts + self.alpha) / (np.sum(counts) + self.alpha * len(counts))  # Suavizado
                    self.feature_probs[cls][feature_idx] = probs
                else:
                    # Caso continuo: calcular media y desviación estándar (asumiendo distribución Gaussiana)
                    mean = np.mean(feature_values)
                    std = np.std(feature_values)
                    self.feature_probs[cls][feature_idx] = (mean, std)  # Guardar parámetros gaussianos

    def _is_discrete(self, values):
        """
        Determina si una característica es discreta.
        Se considera discreta si tiene menos de 10 valores únicos (umbral arbitrario).
        """
        return len(np.unique(values)) < 10

    def _calculate_likelihood(self, x, cls):
        """
        Calcula la probabilidad P(x|cls) para una muestra x y una clase cls.
        x: Vector de características de una muestra.
        cls: Clase para la cual se calcula la probabilidad.
        """
        likelihood = 1.0  # Inicializar la probabilidad como 1 (multiplicativa)
        for feature_idx, value in enumerate(x):
            if isinstance(self.feature_probs[cls][feature_idx], np.ndarray):
                # Caso discreto: obtener la probabilidad del valor actual
                prob = self.feature_probs[cls][feature_idx][int(value)] if int(value) < len(self.feature_probs[cls][feature_idx]) else self.alpha
            else:
                # Caso continuo: calcular probabilidad usando la función de densidad Gaussiana
                mean, std = self.feature_probs[cls][feature_idx]
                prob = np.exp(-((value - mean) ** 2) / (2 * std ** 2)) / (np.sqrt(2 * np.pi) * std)
            likelihood *= prob  # Multiplicar las probabilidades de cada característica
        return likelihood

    def predict(self, X):
        """
        Predice las clases para un conjunto de muestras.
        X: Matriz de características (n_samples, n_features).
        """
        predictions = []  # Lista para almacenar las predicciones
        for x in X:
            posteriors = {}  # Diccionario para almacenar las probabilidades posteriores P(y|x)
            for cls in self.classes:
                prior = np.log(self.class_probs[cls])  # Logaritmo de la probabilidad a priori P(y)
                likelihood = np.log(self._calculate_likelihood(x, cls))  # Logaritmo de la probabilidad P(x|y)
                posteriors[cls] = prior + likelihood  # Sumar logaritmos para evitar underflow
            predictions.append(max(posteriors, key=posteriors.get))  # Seleccionar la clase con mayor probabilidad posterior
        return np.array(predictions)

# Ejemplo con dataset de flores Iris (discreto)
from sklearn.datasets import load_iris
from sklearn.preprocessing import KBinsDiscretizer

# Cargar el dataset Iris
iris = load_iris()
X, y = iris.data, iris.target

# Discretizar características para este ejemplo
discretizer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
X_discrete = discretizer.fit_transform(X)

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X_discrete, y, test_size=0.2)

# Uso del clasificador Naïve Bayes
nb = NaiveBayesClassifier(alpha=1.0)  # Crear una instancia del clasificador
nb.fit(X_train, y_train)  # Entrenar el modelo
predictions = nb.predict(X_test)  # Realizar predicciones

# Evaluar el modelo
print(f"Precisión: {accuracy_score(y_test, predictions):.2f}")  # Imprimir precisión
print("Ejemplo de predicciones:", predictions[:10])  # Mostrar algunas predicciones
print("Valores reales:", y_test[:10])  # Mostrar valores reales correspondientes