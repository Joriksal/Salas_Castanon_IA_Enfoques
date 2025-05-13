# Importación de librerías necesarias
import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos multidimensionales
import matplotlib.pyplot as plt  # Librería para visualización de datos (gráficas)
from sklearn.datasets import make_blobs, make_moons  # Generación de conjuntos de datos de prueba
from sklearn.neighbors import NearestNeighbors  # Algoritmo para encontrar vecinos más cercanos
from collections import Counter  # Herramienta para contar elementos en estructuras de datos
from sklearn.metrics import silhouette_score  # Métrica para evaluar la calidad de los clusters

# Clase para implementar el algoritmo k-Nearest Neighbors (k-NN)
class KNN:
    """Implementación completa de k-Nearest Neighbors (k-NN), un algoritmo de clasificación supervisada."""
    def __init__(self, k=3):
        # Constructor de la clase. Define el número de vecinos a considerar (k).
        self.k = k  # Número de vecinos más cercanos a considerar para la clasificación
    
    def fit(self, X, y):
        # Método para entrenar el modelo. Almacena los datos de entrenamiento.
        self.X_train = X  # Características de los datos de entrenamiento
        self.y_train = y  # Etiquetas de los datos de entrenamiento
    
    def predict(self, X):
        # Predice la clase para cada muestra en X (datos de prueba).
        return np.array([self._predict(x) for x in X])  # Llama al método interno _predict para cada muestra
    
    def _predict(self, x):
        # Método interno para predecir la clase de una sola muestra.
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]  # Calcula la distancia euclidiana entre x y cada punto de entrenamiento
        k_indices = np.argsort(distances)[:self.k]  # Obtiene los índices de los k vecinos más cercanos
        k_labels = [self.y_train[i] for i in k_indices]  # Obtiene las etiquetas de los k vecinos más cercanos
        return Counter(k_labels).most_common(1)[0][0]  # Devuelve la etiqueta más común entre los vecinos

# Clase para implementar el algoritmo k-Medias (k-Means)
class KMeans:
    """Implementación completa de k-Medias, un algoritmo de clustering no supervisado."""
    def __init__(self, k=3, max_iter=100):
        # Constructor de la clase. Define el número de clusters (k) y el número máximo de iteraciones.
        self.k = k  # Número de clusters
        self.max_iter = max_iter  # Número máximo de iteraciones para ajustar los centroides
    
    def fit(self, X):
        # Método para ajustar los clusters a los datos.
        self.centroids = X[np.random.choice(X.shape[0], self.k, replace=False)]  # Inicializa los centroides aleatoriamente
        for _ in range(self.max_iter):  # Itera hasta alcanzar el máximo de iteraciones o convergencia
            clusters = self._create_clusters(X)  # Asigna cada punto al cluster más cercano
            prev_centroids = self.centroids  # Guarda los centroides anteriores
            self.centroids = np.array([X[cluster].mean(axis=0) for cluster in clusters])  # Calcula los nuevos centroides
            if np.allclose(prev_centroids, self.centroids):  # Si los centroides no cambian, termina
                break
    
    def _create_clusters(self, X):
        # Método interno para asignar puntos a clusters.
        clusters = [[] for _ in range(self.k)]  # Crea una lista de clusters vacíos
        for idx, sample in enumerate(X):  # Itera sobre cada muestra
            centroid_idx = np.argmin([np.linalg.norm(sample - c) for c in self.centroids])  # Encuentra el índice del centroide más cercano
            clusters[centroid_idx].append(idx)  # Asigna el punto al cluster correspondiente
        return clusters
    
    def predict(self, X):
        # Predice el cluster para cada punto en X.
        return np.array([np.argmin([np.linalg.norm(x - c) for c in self.centroids]) for x in X])

# Función para visualizar clusters
def plot_clusters(X, y, title):
    """Visualización de clusters en un gráfico 2D."""
    plt.figure(figsize=(10, 6))  # Define el tamaño de la figura
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap='viridis')  # Crea un gráfico de dispersión con colores según los clusters
    plt.title(title)  # Título del gráfico
    plt.show()  # Muestra el gráfico

# 1. Generación de datos de prueba
X_blobs, y_blobs = make_blobs(n_samples=300, centers=3, random_state=42)  # Genera datos con 3 clusters
X_moons, _ = make_moons(n_samples=300, noise=0.05, random_state=42)  # Genera datos con forma de luna

# 2. k-NN (Clasificación)
knn = KNN(k=5)  # Instancia de k-NN con k=5
knn.fit(X_blobs, y_blobs)  # Entrena el modelo con los datos
y_pred_knn = knn.predict(X_blobs)  # Predice las etiquetas
plot_clusters(X_blobs, y_pred_knn, "k-NN Classification")  # Visualiza los resultados

# 3. k-Means (Clustering)
kmeans = KMeans(k=3)  # Instancia de k-Medias con k=3
kmeans.fit(X_blobs)  # Ajusta los clusters
y_pred_kmeans = kmeans.predict(X_blobs)  # Predice los clusters
plot_clusters(X_blobs, y_pred_kmeans, "k-Means Clustering")  # Visualiza los resultados

# 4. DBSCAN (Clustering basado en densidad)
def dbscan(X, eps=0.5, min_samples=5):
    """
    Implementación simplificada de DBSCAN (Density-Based Spatial Clustering of Applications with Noise).
    """
    neigh = NearestNeighbors(n_neighbors=min_samples)  # Encuentra vecinos cercanos
    neigh.fit(X)  # Ajusta el modelo a los datos
    distances = np.sort(neigh.kneighbors()[0][:, -1])  # Distancias al vecino más lejano
    
    # Determinación automática de épsilon (opcional)
    if eps == 'auto':
        eps = distances[int(len(distances)*0.95)]  # Percentil 95
    
    # Inicializa etiquetas (-1 para ruido)
    labels = np.zeros(X.shape[0]) - 1
    cluster_id = 0  # ID del cluster
    
    for i in range(X.shape[0]):
        if labels[i] != -1:  # Si ya está etiquetado, continúa
            continue
            
        # Encuentra vecinos dentro del radio eps
        neighbors = np.where(np.linalg.norm(X - X[i], axis=1) < eps)[0]
        
        if len(neighbors) < min_samples:  # Si no cumple el mínimo, es ruido
            continue
            
        # Expande el cluster
        labels[neighbors] = cluster_id
        queue = list(neighbors)
        
        while queue:
            current = queue.pop()
            new_neighbors = np.where(np.linalg.norm(X - X[current], axis=1) < eps)[0]
            
            if len(new_neighbors) >= min_samples:
                for n in new_neighbors:
                    if labels[n] == -1:  # Si es ruido, lo añade al cluster
                        labels[n] = cluster_id
                        queue.append(n)
        
        cluster_id += 1  # Incrementa el ID del cluster
    
    return labels

y_pred_dbscan = dbscan(X_moons, eps=0.2, min_samples=5)  # Aplica DBSCAN
plot_clusters(X_moons, y_pred_dbscan, "DBSCAN Clustering")  # Visualiza los resultados

# 5. Evaluación comparativa de los métodos
print("\nMétricas de evaluación:")
print(f"k-Means Silhouette: {silhouette_score(X_blobs, y_pred_kmeans):.2f}")  # Métrica de Silhouette para k-Means
print(f"DBSCAN Silhouette: {silhouette_score(X_moons, y_pred_dbscan):.2f}")  # Métrica de Silhouette para DBSCAN

# 6. Método del codo para determinar el mejor k en k-Means
wcss = []  # Lista para almacenar la inercia (WCSS)
k_range = range(1, 11)  # Rango de valores de k

for k in k_range:
    kmeans = KMeans(k=k)  # Instancia de k-Medias con k variable
    kmeans.fit(X_blobs)  # Ajusta los clusters
    # Calcula la suma de las distancias cuadradas dentro de los clusters
    wcss.append(np.sum([np.sum((X_blobs[kmeans.predict(X_blobs) == i] - kmeans.centroids[i])**2) 
                 for i in range(k)]))

# Grafica el método del codo
plt.figure(figsize=(10, 6))
plt.plot(k_range, wcss, 'bo-')
plt.title('Método del Codo para k-Means')
plt.xlabel('Número de clusters (k)')
plt.ylabel('WCSS (Inercia)')
plt.show()