# Importa la librería NumPy, que se utiliza para realizar operaciones matemáticas y manipulación de arreglos.
import numpy as np

# Importa Matplotlib para la generación de gráficos. Aquí se usa específicamente pyplot para crear gráficos 2D.
import matplotlib.pyplot as plt

# Importa GaussianMixture de sklearn.mixture, que permite trabajar con modelos de mezcla gaussiana (GMM).
from sklearn.mixture import GaussianMixture

# Importa make_blobs de sklearn.datasets, que se utiliza para generar datos sintéticos en forma de clusters.
from sklearn.datasets import make_blobs

# Clase para segmentar clientes utilizando un modelo de mezcla gaussiana
class CustomerSegmenter:
    def __init__(self, n_components=3):
        """
        Constructor de la clase CustomerSegmenter.
        Inicializa un modelo de mezcla gaussiana (GaussianMixture) con el número de componentes especificado.
        
        Parámetros:
        - n_components: Número de clusters o componentes gaussianos que se ajustarán a los datos.
        """
        self.gmm = GaussianMixture(
            n_components=n_components,  # Número de componentes gaussianos
            covariance_type='full',     # Tipo de covarianza ('full' permite covarianzas completas)
            random_state=42             # Semilla para reproducibilidad
        )
        self.fitted = False  # Bandera para indicar si el modelo ha sido entrenado

    def fit(self, X):
        """
        Ajusta el modelo de mezcla gaussiana a los datos proporcionados.
        
        Parámetros:
        - X: Datos de entrada (matriz de características).
        """
        self.gmm.fit(X)  # Entrena el modelo con los datos
        self.fitted = True  # Marca el modelo como entrenado

    def predict_proba(self, X):
        """
        Calcula las probabilidades de pertenencia a cada cluster para los datos proporcionados.
        
        Parámetros:
        - X: Datos de entrada para los que se calcularán las probabilidades.
        
        Retorna:
        - Matriz de probabilidades, donde cada fila corresponde a un dato y cada columna a un cluster.
        """
        if not self.fitted:
            raise ValueError("Modelo no entrenado. Llame primero a fit()")
        return self.gmm.predict_proba(X)  # Calcula las probabilidades de pertenencia

    def sample_from_distribution(self, n_samples=1):
        """
        Genera muestras sintéticas basadas en las distribuciones gaussianas ajustadas.
        
        Parámetros:
        - n_samples: Número de muestras a generar.
        
        Retorna:
        - Muestras generadas como un arreglo NumPy.
        """
        if not self.fitted:
            raise ValueError("Modelo no entrenado. Llame primero a fit()")
        return self.gmm.sample(n_samples)[0]  # Genera muestras sintéticas

    def plot_distributions(self, X):
        """
        Genera un gráfico que muestra las distribuciones gaussianas ajustadas y los datos originales.
        
        Parámetros:
        - X: Datos originales utilizados para ajustar el modelo.
        """
        if not self.fitted:
            raise ValueError("Modelo no entrenado. Llame primero a fit()")
        
        plt.figure(figsize=(10, 6))  # Configura el tamaño del gráfico
        plt.scatter(X[:, 0], X[:, 1], c='gray', alpha=0.5, label='Datos')  # Dibuja los datos originales
        
        means = self.gmm.means_  # Obtiene las medias de las distribuciones gaussianas
        plt.scatter(means[:, 0], means[:, 1], c='red', s=100, marker='X', label='Medias')  # Dibuja las medias
        
        # Dibuja las elipses que representan las covarianzas de las distribuciones
        for i, (mean, cov) in enumerate(zip(means, self.gmm.covariances_)):
            v, w = np.linalg.eigh(cov)  # Descomposición en valores y vectores propios
            angle = np.degrees(np.arctan2(w[0][1], w[0][0]))  # Calcula el ángulo de rotación
            v = 2. * np.sqrt(2.) * np.sqrt(v)  # Escala los valores propios
            ell = plt.matplotlib.patches.Ellipse(
                xy=mean,  # Centro de la elipse
                width=v[0],  # Ancho basado en el valor propio mayor
                height=v[1],  # Alto basado en el valor propio menor
                angle=180 + angle,  # Ángulo de rotación
                color=f'C{i}',  # Color único para cada cluster
                alpha=0.3  # Transparencia
            )
            plt.gca().add_artist(ell)  # Añade la elipse al gráfico
        
        plt.gca().set_aspect('equal', 'datalim')  # Ajusta la escala del gráfico
        plt.title('Distribuciones Gaussianas Ajustadas')  # Título del gráfico
        plt.xlabel('Gasto Mensual (normalizado)')  # Etiqueta del eje X
        plt.ylabel('Frecuencia de Compra (normalizado)')  # Etiqueta del eje Y
        plt.legend()  # Muestra la leyenda
        plt.grid(True)  # Activa la cuadrícula
        plt.show()  # Muestra el gráfico

# Ejemplo de uso
if __name__ == "__main__":
    # Genera datos sintéticos para simular clientes
    np.random.seed(42)  # Fija la semilla para reproducibilidad
    X, _ = make_blobs(
        n_samples=500,  # Número de muestras
        centers=3,      # Número de clusters
        cluster_std=1.5,  # Desviación estándar de los clusters
        n_features=2,   # Número de características (dimensiones)
        random_state=42  # Semilla para reproducibilidad
    )
    # Normaliza los datos para que tengan media 0 y desviación estándar 1
    X = (X - X.mean(axis=0)) / X.std(axis=0)
    
    # Crea una instancia del segmentador de clientes
    segmenter = CustomerSegmenter(n_components=3)
    # Ajusta el modelo a los datos
    segmenter.fit(X)
    # Genera un gráfico de las distribuciones ajustadas
    segmenter.plot_distributions(X)
    
    # Define algunas muestras de prueba para calcular probabilidades de pertenencia
    test_samples = np.array([[1.5, 0.8], [-1, -1], [0, 1.2]])  # Muestras de prueba
    probabilities = segmenter.predict_proba(test_samples)  # Calcula las probabilidades
    
    # Imprime las probabilidades de pertenencia a cada cluster para las muestras de prueba
    print("\nProbabilidades de pertenencia a clusters:")
    for i, probs in enumerate(probabilities):
        print(f"Muestra {i+1}: {np.round(probs, 3)}")
    
    # Genera clientes sintéticos basados en las distribuciones ajustadas
    synthetic_customers = segmenter.sample_from_distribution(5)  # Genera 5 muestras
    print("\nClientes sintéticos generados:")
    print(np.round(synthetic_customers, 2))  # Imprime las muestras generadas