import numpy as np  # Para realizar operaciones matemáticas avanzadas y manejo eficiente de arreglos multidimensionales.
from sklearn.tree import DecisionTreeRegressor  # Para implementar árboles de decisión para tareas de regresión.
from sklearn.linear_model import LinearRegression  # Para realizar regresión lineal en las hojas del árbol M5.
from sklearn.base import BaseEstimator, RegressorMixin  # Para crear un estimador personalizado compatible con scikit-learn.

class M5RegressionTree(BaseEstimator, RegressorMixin):
    """Implementación personalizada del árbol M5 para regresión.
    
    Este modelo combina árboles de decisión con regresión lineal en las hojas.
    """

    def __init__(self, max_depth=5, min_samples_split=2):
        """
        Inicializa el árbol M5 con parámetros específicos.

        Args:
            max_depth (int): Profundidad máxima del árbol.
            min_samples_split (int): Número mínimo de muestras para dividir un nodo.
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.tree = None

    def _split_data(self, X, y, feature_idx, threshold):
        """
        Divide los datos en dos subconjuntos según un umbral en una característica.

        Args:
            X (ndarray): Matriz de características.
            y (ndarray): Vector de etiquetas.
            feature_idx (int): Índice de la característica a dividir.
            threshold (float): Umbral para la división.

        Returns:
            tuple: Subconjuntos de características y etiquetas (X_left, y_left, X_right, y_right).
        """
        left_mask = X[:, feature_idx] <= threshold
        right_mask = ~left_mask
        return X[left_mask], y[left_mask], X[right_mask], y[right_mask]

    def _fit_node(self, X, y, depth):
        """
        Construye el árbol recursivamente. En las hojas, ajusta un modelo de regresión lineal.

        Args:
            X (ndarray): Matriz de características.
            y (ndarray): Vector de etiquetas.
            depth (int): Profundidad actual del nodo.

        Returns:
            dict: Nodo del árbol (puede ser una hoja o un nodo interno).
        """
        # Condición de parada: profundidad máxima o pocas muestras
        if depth >= self.max_depth or len(X) < self.min_samples_split:
            model = LinearRegression().fit(X, y)  # Ajusta un modelo lineal en la hoja
            return {'model': model, 'is_leaf': True}

        # Inicializa las variables para encontrar la mejor división
        best_feature, best_threshold = None, None
        best_mse = float('inf')  # Error cuadrático medio mínimo

        # Busca la mejor característica y umbral para dividir
        for feature_idx in range(X.shape[1]):
            thresholds = np.unique(X[:, feature_idx])  # Umbrales únicos de la característica
            for threshold in thresholds:
                # Divide los datos según el umbral
                X_left, y_left, X_right, y_right = self._split_data(X, y, feature_idx, threshold)
                if len(y_left) == 0 or len(y_right) == 0:
                    continue  # Ignora divisiones inválidas

                # Calcula el MSE ponderado de la división
                mse = (np.mean((y_left - np.mean(y_left))**2) * len(y_left) +
                       np.mean((y_right - np.mean(y_right))**2) * len(y_right))
                if mse < best_mse:
                    best_mse = mse
                    best_feature = feature_idx
                    best_threshold = threshold

        # Si no se encuentra una división válida, convierte el nodo en una hoja
        if best_feature is None:
            model = LinearRegression().fit(X, y)
            return {'model': model, 'is_leaf': True}

        # Divide los datos usando la mejor característica y umbral
        X_left, y_left, X_right, y_right = self._split_data(X, y, best_feature, best_threshold)
        # Crea un nodo interno con referencias a los nodos hijos
        node = {
            'feature_idx': best_feature,
            'threshold': best_threshold,
            'is_leaf': False,
            'left': self._fit_node(X_left, y_left, depth + 1),
            'right': self._fit_node(X_right, y_right, depth + 1)
        }
        return node

    def fit(self, X, y):
        """
        Entrena el árbol M5 ajustando los nodos recursivamente.

        Args:
            X (ndarray): Matriz de características.
            y (ndarray): Vector de etiquetas.

        Returns:
            self: El modelo ajustado.
        """
        self.tree = self._fit_node(X, y, 0)  # Construye el árbol desde la raíz
        return self

    def _predict_node(self, x, node):
        """
        Predice un valor recorriendo el árbol desde un nodo específico.

        Args:
            x (ndarray): Vector de características de una muestra.
            node (dict): Nodo actual del árbol.

        Returns:
            float: Predicción para la muestra.
        """
        if node['is_leaf']:
            return node['model'].predict(x.reshape(1, -1))[0]  # Predicción de la hoja
        if x[node['feature_idx']] <= node['threshold']:
            return self._predict_node(x, node['left'])  # Baja por el subárbol izquierdo
        else:
            return self._predict_node(x, node['right'])  # Baja por el subárbol derecho

    def predict(self, X):
        """
        Predice valores para un conjunto de datos.

        Args:
            X (ndarray): Matriz de características.

        Returns:
            ndarray: Vector de predicciones.
        """
        return np.array([self._predict_node(x, self.tree) for x in X])

# --- Ejemplo práctico ---
if __name__ == "__main__":
    # Dataset de ejemplo: [Característica1, Característica2], Target
    X = np.array([[1, 2], [2, 3], [3, 4], [4, 5], [5, 6]])
    y = np.array([2.1, 3.9, 6.2, 8.1, 9.8])  # y ≈ Característica1 + Característica2

    # Entrenamiento del modelo
    m5_tree = M5RegressionTree(max_depth=3)
    m5_tree.fit(X, y)

    # Predicción con nuevos datos
    X_test = np.array([[2.5, 3.5], [4.5, 5.5]])
    print("Predicciones M5:", m5_tree.predict(X_test))  # Ejemplo: [5.0, 9.5]