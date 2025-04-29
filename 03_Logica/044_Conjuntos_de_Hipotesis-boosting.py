import numpy as np
from sklearn.tree import DecisionTreeClassifier

class AdaBoost:
    """Implementación de AdaBoost para clasificación binaria."""
    
    def __init__(self, n_estimators=50):
        """
        Inicializa el modelo AdaBoost.
        
        Parámetros:
        - n_estimators: Número de estimadores débiles (weak learners) a entrenar.
        """
        self.n_estimators = n_estimators
        self.estimators = []  # Lista para almacenar los estimadores débiles
        self.estimator_weights = []  # Lista para almacenar los pesos de cada estimador
        
    def fit(self, X, y):
        """
        Entrena el modelo AdaBoost en el conjunto de datos dado.
        
        Parámetros:
        - X: Matriz de características (n_samples x n_features).
        - y: Vector de etiquetas binarias (-1, 1).
        """
        n_samples = X.shape[0]  # Número de muestras
        weights = np.ones(n_samples) / n_samples  # Inicializar pesos uniformes para las muestras
        
        for _ in range(self.n_estimators):
            # Entrenar un estimador débil (árbol de decisión de profundidad 1)
            estimator = DecisionTreeClassifier(max_depth=1)
            estimator.fit(X, y, sample_weight=weights)  # Entrenar usando los pesos actuales
            predictions = estimator.predict(X)  # Predicciones del estimador
            
            # Calcular el error ponderado del estimador
            error = np.sum(weights * (predictions != y)) / np.sum(weights)
            
            # Calcular el peso del estimador (alpha)
            # alpha mide la importancia del estimador basado en su precisión
            alpha = 0.5 * np.log((1 - error) / (error + 1e-10))  # Evitar división por cero
            
            # Actualizar los pesos de las muestras
            # Las muestras mal clasificadas reciben mayor peso
            weights *= np.exp(-alpha * y * predictions)
            weights /= np.sum(weights)  # Normalizar los pesos para que sumen 1
            
            # Guardar el estimador y su peso
            self.estimators.append(estimator)
            self.estimator_weights.append(alpha)
    
    def predict(self, X):
        """
        Realiza predicciones para las muestras dadas.
        
        Parámetros:
        - X: Matriz de características (n_samples x n_features).
        
        Retorna:
        - Vector de predicciones (-1, 1) para cada muestra.
        """
        # Calcular la suma ponderada de las predicciones de todos los estimadores
        predictions = np.array([
            alpha * estimator.predict(X)  # Ponderar las predicciones por el peso del estimador
            for estimator, alpha in zip(self.estimators, self.estimator_weights)
        ])
        # La clase final es el signo de la suma ponderada
        return np.sign(np.sum(predictions, axis=0))

# --- Ejemplo práctico ---
if __name__ == "__main__":
    # Dataset de ejemplo (X: características, y: etiquetas binarias [-1, 1])
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6], [6, 7]])  # Datos de entrada
    y = np.array([-1, -1, 1, 1, 1, -1])  # Etiquetas correspondientes (no linealmente separables)
    
    # Crear y entrenar el modelo AdaBoost
    adaboost = AdaBoost(n_estimators=10)  # Usar 10 estimadores débiles
    adaboost.fit(X, y)  # Entrenar el modelo
    
    # Predicción en nuevos datos
    X_test = np.array([[2.5, 3.5], [5.5, 6.5]])  # Nuevas muestras para clasificar
    print("Predicciones AdaBoost:", adaboost.predict(X_test))  # Ejemplo de salida: [-1, 1]