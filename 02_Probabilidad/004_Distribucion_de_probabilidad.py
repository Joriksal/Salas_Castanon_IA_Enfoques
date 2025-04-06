import numpy as np
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs

# Clase para segmentar clientes utilizando un modelo de mezcla gaussiana
class CustomerSegmenter:
    def __init__(self, n_components=3):
        # Inicializa el modelo de mezcla gaussiana con el número de componentes especificado
        self.gmm = GaussianMixture(n_components=n_components, 
                                   covariance_type='full', 
                                   random_state=42)
        self.fitted = False  # Indica si el modelo ha sido entrenado
    
    def fit(self, X):
        # Ajusta el modelo a los datos proporcionados
        self.gmm.fit(X)
        self.fitted = True  # Marca el modelo como entrenado
    
    def predict_proba(self, X):
        # Calcula las probabilidades de pertenencia a cada cluster para los datos proporcionados
        if not self.fitted:
            raise ValueError("Modelo no entrenado. Llame primero a fit()")
        return self.gmm.predict_proba(X)
    
    def sample_from_distribution(self, n_samples=1):
        # Genera muestras sintéticas basadas en la distribución ajustada
        if not self.fitted:
            raise ValueError("Modelo no entrenado. Llame primero a fit()")
        return self.gmm.sample(n_samples)[0]
    
    def plot_distributions(self, X):
        # Genera un gráfico de las distribuciones gaussianas ajustadas
        if not self.fitted:
            raise ValueError("Modelo no entrenado. Llame primero a fit()")
            
        plt.figure(figsize=(10, 6))
        # Dibuja los datos originales en el gráfico
        plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, label='Datos')
        
        # Obtiene las medias de las distribuciones gaussianas
        means = self.gmm.means_
        # Dibuja las medias en el gráfico
        plt.scatter(means[:, 0], means[:, 1], c='red', s=100, 
                    marker='X', label='Medias')
        
        # Dibuja las elipses que representan las covarianzas de las distribuciones
        for i, (mean, cov) in enumerate(zip(means, self.gmm.covariances_)):
            v, w = np.linalg.eigh(cov)  # Descomposición en valores y vectores propios
            angle = np.degrees(np.arctan2(w[0][1], w[0][0]))  # Calcula el ángulo de rotación
            v = 2. * np.sqrt(2.) * np.sqrt(v)  # Escala los valores propios
            # Crea una elipse para representar la distribución
            ell = plt.matplotlib.patches.Ellipse(
                xy=mean,
                width=v[0],
                height=v[1],
                angle=180 + angle,
                color=f'C{i}',
                alpha=0.3
            )
            plt.gca().add_artist(ell)  # Añade la elipse al gráfico
        
        plt.gca().set_aspect('equal', 'datalim')  # Ajusta la escala del gráfico
        plt.title('Distribuciones Gaussianas Ajustadas')
        plt.xlabel('Gasto Mensual (normalizado)')
        plt.ylabel('Frecuencia de Compra (normalizado)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Ejemplo de uso
if __name__ == "__main__":
    # Genera datos sintéticos para simular clientes
    np.random.seed(42)
    X, _ = make_blobs(n_samples=500, centers=3, cluster_std=1.5,
                      n_features=2, random_state=42)
    # Normaliza los datos para que tengan media 0 y desviación estándar 1
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Crea una instancia del segmentador de clientes
    segmenter = CustomerSegmenter(n_components=3)
    # Ajusta el modelo a los datos
    segmenter.fit(X)
    # Genera un gráfico de las distribuciones ajustadas
    segmenter.plot_distributions(X)
    
    # Define algunas muestras de prueba para calcular probabilidades de pertenencia
    test_samples = np.array([[1.5, 0.8], [-1, -1], [0, 1.2]])
    probabilities = segmenter.predict_proba(test_samples)
    
    # Imprime las probabilidades de pertenencia a cada cluster para las muestras de prueba
    print("\nProbabilidades de pertenencia a clusters:")
    for i, probs in enumerate(probabilities):
        print(f"Muestra {i+1}: {np.round(probs, 3)}")
    
    # Genera clientes sintéticos basados en las distribuciones ajustadas
    synthetic_customers = segmenter.sample_from_distribution(5)
    print("\nClientes sintéticos generados:")
    print(np.round(synthetic_customers, 2))