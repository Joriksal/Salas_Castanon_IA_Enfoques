# Importamos las librerías necesarias
import numpy as np  # Biblioteca para cálculos numéricos y manejo de arreglos
import matplotlib.pyplot as plt  # Biblioteca para visualización de datos
from sklearn.datasets import make_moons  # Generador de datos no lineales en forma de lunas
from sklearn.model_selection import train_test_split  # Para dividir los datos en conjuntos de entrenamiento y prueba
from sklearn.metrics import accuracy_score  # Para calcular la precisión del modelo

# Definimos la clase SVM (Máquina de Vectores de Soporte)
class SVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3):
        """
        Constructor de la clase SVM.
        Parámetros:
        - kernel: Tipo de kernel a usar ('linear', 'poly', 'rbf').
        - C: Parámetro de regularización que controla el margen.
        - gamma: Parámetro para los kernels RBF y polinomial.
        - degree: Grado del polinomio para el kernel polinomial.
        """
        self.kernel = kernel  # Tipo de kernel
        self.C = C  # Parámetro de regularización
        self.gamma = gamma  # Parámetro para kernels no lineales
        self.degree = degree  # Grado del kernel polinomial

    def _kernel_function(self, x1, x2):
        """
        Calcula la función kernel entre dos vectores x1 y x2.
        """
        if self.kernel == 'linear':  # Kernel lineal
            return np.dot(x1, x2)  # Producto punto entre x1 y x2
        elif self.kernel == 'poly':  # Kernel polinomial
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':  # Kernel RBF (Radial Basis Function)
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)  # Exponencial de la distancia euclidiana al cuadrado

    def fit(self, X, y, max_iter=1000, tol=1e-3):
        """
        Entrena el modelo SVM utilizando el algoritmo SMO (Sequential Minimal Optimization).
        Parámetros:
        - X: Datos de entrada (matriz de características).
        - y: Etiquetas de los datos (0 o 1).
        - max_iter: Número máximo de iteraciones.
        - tol: Tolerancia para la convergencia.
        """
        n_samples, n_features = X.shape  # Número de muestras y características
        self.X = X  # Guardamos los datos de entrada
        self.y = y * 2 - 1  # Convertimos las etiquetas a -1 y 1 (requisito para SVM)

        # Inicializamos los parámetros
        self.alpha = np.zeros(n_samples)  # Multiplicadores de Lagrange
        self.b = 0.0  # Sesgo

        # Calculamos la matriz kernel
        K = np.zeros((n_samples, n_samples))  # Matriz de tamaño (n_samples x n_samples)
        for i in range(n_samples):
            for j in range(n_samples):
                K[i, j] = self._kernel_function(X[i], X[j])  # Aplicamos la función kernel

        # Algoritmo SMO para optimizar los multiplicadores de Lagrange
        for _ in range(max_iter):
            alpha_prev = np.copy(self.alpha)  # Guardamos los valores anteriores de alpha

            for i in range(n_samples):
                # Calculamos el error para la muestra i
                Ei = self._decision_function(X[i]) - self.y[i]

                # Seleccionamos un índice j diferente de i
                j = np.random.choice([x for x in range(n_samples) if x != i])
                Ej = self._decision_function(X[j]) - self.y[j]

                # Guardamos los valores antiguos de alpha[i] y alpha[j]
                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]

                # Calculamos los límites L y H para alpha[j]
                if self.y[i] == self.y[j]:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                else:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])

                if L == H:  # Si los límites son iguales, no actualizamos
                    continue

                # Calculamos eta (la segunda derivada del problema dual)
                eta = 2 * K[i, j] - K[i, i] - K[j, j]
                if eta >= 0:  # Si eta es no negativa, no actualizamos
                    continue

                # Actualizamos alpha[j]
                self.alpha[j] -= (self.y[j] * (Ei - Ej)) / eta

                # Limitamos alpha[j] dentro de los límites L y H
                self.alpha[j] = np.clip(self.alpha[j], L, H)

                # Actualizamos alpha[i]
                self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])

                # Calculamos el sesgo b
                b1 = self.b - Ei - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, i] \
                     - self.y[j] * (self.alpha[j] - alpha_j_old) * K[i, j]
                b2 = self.b - Ej - self.y[i] * (self.alpha[i] - alpha_i_old) * K[i, j] \
                     - self.y[j] * (self.alpha[j] - alpha_j_old) * K[j, j]

                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2

            # Verificamos la convergencia
            if np.linalg.norm(self.alpha - alpha_prev) < tol:
                break

    def _decision_function(self, x):
        """
        Calcula la función de decisión para una muestra x.
        """
        result = self.b  # Inicializamos con el sesgo
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
        return result

    def predict(self, X):
        """
        Realiza predicciones para un conjunto de datos X.
        """
        return np.array([1 if self._decision_function(x) >= 0 else 0 for x in X])

# Generamos datos no lineales en forma de lunas
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)  # 100 muestras con ruido
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)  # Dividimos en entrenamiento y prueba

# Entrenamos el modelo SVM con kernel RBF
svm = SVM(kernel='rbf', C=1.0, gamma=10)  # Creamos una instancia de SVM
svm.fit(X_train, y_train)  # Entrenamos el modelo

# Evaluamos el modelo
y_pred = svm.predict(X_test)  # Realizamos predicciones
print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")  # Calculamos la precisión

# Visualizamos la frontera de decisión
def plot_decision_boundary(model, X, y):
    """
    Grafica la frontera de decisión del modelo.
    """
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5  # Límites del eje x
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5  # Límites del eje y
    h = 0.02  # Tamaño del paso para la malla

    # Creamos una malla de puntos
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # Predecimos para cada punto de la malla
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)  # Damos forma a los resultados

    # Graficamos la frontera de decisión
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)  # Frontera de decisión
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')  # Puntos de datos
    plt.title(f"SVM con kernel {model.kernel}")
    plt.show()

plot_decision_boundary(svm, X, y)  # Llamamos a la función para graficar