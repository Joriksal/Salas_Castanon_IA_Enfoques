import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

class SVM:
    def __init__(self, kernel='rbf', C=1.0, gamma=1.0, degree=3):
        self.kernel = kernel
        self.C = C          # Parámetro de regularización
        self.gamma = gamma  # Para RBF/polinomial
        self.degree = degree  # Para kernel polinomial
        
    def _kernel_function(self, x1, x2):
        """Calcula la función kernel"""
        if self.kernel == 'linear':
            return np.dot(x1, x2)
        elif self.kernel == 'poly':
            return (self.gamma * np.dot(x1, x2) + 1) ** self.degree
        elif self.kernel == 'rbf':
            return np.exp(-self.gamma * np.linalg.norm(x1 - x2)**2)
    
    def fit(self, X, y, max_iter=1000, tol=1e-3):
        """Entrenamiento con SMO (Sequential Minimal Optimization)"""
        n_samples, n_features = X.shape
        self.X = X
        self.y = y * 2 - 1  # Convertir a -1, 1
        
        # Inicializar parámetros
        self.alpha = np.zeros(n_samples)
        self.b = 0.0
        
        # Matriz kernel
        K = np.zeros((n_samples, n_samples))
        for i in range(n_samples):
            for j in range(n_samples):
                K[i,j] = self._kernel_function(X[i], X[j])
        
        # Optimización SMO
        for _ in range(max_iter):
            alpha_prev = np.copy(self.alpha)
            
            for i in range(n_samples):
                # Cálculo de error
                Ei = self._decision_function(X[i]) - self.y[i]
                
                # Selección de j ≠ i
                j = np.random.choice([x for x in range(n_samples) if x != i])
                Ej = self._decision_function(X[j]) - self.y[j]
                
                # Guardar valores antiguos
                alpha_i_old, alpha_j_old = self.alpha[i], self.alpha[j]
                
                # Calcular L y H (restricciones de caja)
                if self.y[i] == self.y[j]:
                    L = max(0, self.alpha[i] + self.alpha[j] - self.C)
                    H = min(self.C, self.alpha[i] + self.alpha[j])
                else:
                    L = max(0, self.alpha[j] - self.alpha[i])
                    H = min(self.C, self.C + self.alpha[j] - self.alpha[i])
                
                if L == H:
                    continue
                
                # Calcular eta
                eta = 2 * K[i,j] - K[i,i] - K[j,j]
                if eta >= 0:
                    continue
                
                # Actualizar alpha[j]
                self.alpha[j] -= (self.y[j] * (Ei - Ej)) / eta
                
                # Clip alpha[j]
                self.alpha[j] = np.clip(self.alpha[j], L, H)
                
                # Actualizar alpha[i]
                self.alpha[i] += self.y[i] * self.y[j] * (alpha_j_old - self.alpha[j])
                
                # Calcular b
                b1 = self.b - Ei - self.y[i]*(self.alpha[i]-alpha_i_old)*K[i,i] \
                     - self.y[j]*(self.alpha[j]-alpha_j_old)*K[i,j]
                b2 = self.b - Ej - self.y[i]*(self.alpha[i]-alpha_i_old)*K[i,j] \
                     - self.y[j]*(self.alpha[j]-alpha_j_old)*K[j,j]
                
                if 0 < self.alpha[i] < self.C:
                    self.b = b1
                elif 0 < self.alpha[j] < self.C:
                    self.b = b2
                else:
                    self.b = (b1 + b2) / 2
            
            # Verificar convergencia
            if np.linalg.norm(self.alpha - alpha_prev) < tol:
                break
    
    def _decision_function(self, x):
        """Función de decisión"""
        result = self.b
        for i in range(len(self.alpha)):
            result += self.alpha[i] * self.y[i] * self._kernel_function(self.X[i], x)
        return result
    
    def predict(self, X):
        """Predicción"""
        return np.array([1 if self._decision_function(x) >= 0 else 0 for x in X])

# Generar datos no lineales (lunas)
X, y = make_moons(n_samples=100, noise=0.15, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Entrenar SVM con kernel RBF
svm = SVM(kernel='rbf', C=1.0, gamma=10)
svm.fit(X_train, y_train)

# Evaluación
y_pred = svm.predict(X_test)
print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# Visualización
def plot_decision_boundary(model, X, y):
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    h = 0.02  # step size
    
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    
    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolors='k')
    plt.title(f"SVM con kernel {model.kernel}")
    plt.show()

plot_decision_boundary(svm, X, y)