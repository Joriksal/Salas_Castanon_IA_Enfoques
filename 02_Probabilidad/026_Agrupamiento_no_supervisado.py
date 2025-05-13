# Importación de librerías necesarias
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# 1. Generación de datos sintéticos (3 grupos claros + ruido)
np.random.seed(42)  # Fijamos la semilla para reproducibilidad
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)  # Creamos 3 clusters
X = np.vstack([X, np.random.uniform(low=-10, high=10, size=(20, 2))])  # Añadimos ruido (puntos aleatorios)

# 2. Preprocesamiento (normalización)
# Escalamos los datos para que tengan media 0 y desviación estándar 1
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Método del codo para determinar k óptimo
# Inicializamos listas para almacenar métricas
wcss = []  # Within-Cluster Sum of Squares (inercia)
silhouettes = []  # Coeficiente Silhouette
k_range = range(2, 8)  # Rango de valores de k a evaluar

# Iteramos sobre diferentes valores de k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42)  # Inicializamos K-Means con k clusters
    kmeans.fit(X_scaled)  # Ajustamos el modelo a los datos
    wcss.append(kmeans.inertia_)  # Guardamos la inercia (suma de distancias al centroide)
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))  # Calculamos el coeficiente Silhouette

# 4. Gráfica doble (Codo + Silhouette)
plt.figure(figsize=(12, 5))

# Gráfica del método del codo
plt.subplot(1, 2, 1)
plt.plot(k_range, wcss, 'bo-')  # Gráfica de inercia vs número de clusters
plt.xlabel('Número de clusters (k)')
plt.ylabel('WCSS (Inercia)')
plt.title('Método del Codo')

# Gráfica del coeficiente Silhouette
plt.subplot(1, 2, 2)
plt.plot(k_range, silhouettes, 'go-')  # Gráfica de Silhouette vs número de clusters
plt.xlabel('Número de clusters (k)')
plt.ylabel('Coeficiente Silhouette')
plt.title('Análisis Silhouette')

plt.tight_layout()  # Ajustamos el diseño de las gráficas
plt.show()

# 5. Agrupamiento final con k óptimo (k=3 según gráficas)
optimal_k = 3  # Seleccionamos el número óptimo de clusters basado en las gráficas
kmeans = KMeans(n_clusters=optimal_k, random_state=42)  # Inicializamos K-Means con k=3
clusters = kmeans.fit_predict(X_scaled)  # Ajustamos el modelo y obtenemos las etiquetas de los clusters

# 6. Visualización profesional de resultados
plt.figure(figsize=(10, 6))
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']  # Colores para los clusters

# Graficar puntos por cluster
for i in range(optimal_k):
    plt.scatter(X_scaled[clusters == i, 0], X_scaled[clusters == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

# Graficar centroides
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, marker='*', c='black', label='Centroides')

# Graficar ruido (si se detectara)
if -1 in clusters:  # Si hay puntos etiquetados como ruido
    plt.scatter(X_scaled[clusters == -1, 0], X_scaled[clusters == -1, 1],
                s=50, c='gray', label='Ruido')

plt.title('Agrupamiento de Datos con K-Means\n(k={})'.format(optimal_k), fontsize=14)
plt.xlabel('Característica 1 (normalizada)', fontsize=12)
plt.ylabel('Característica 2 (normalizada)', fontsize=12)
plt.legend()  # Añadimos leyenda
plt.grid(True, alpha=0.3)  # Añadimos una cuadrícula con transparencia
plt.show()

# 7. Métricas de evaluación
# Mostramos métricas clave para evaluar el modelo
print(f"\nMétricas para k={optimal_k}:")
print(f"- Inercia: {kmeans.inertia_:.2f}")  # Inercia final
print(f"- Silhouette Score: {silhouette_score(X_scaled, clusters):.2f}")  # Coeficiente Silhouette
print("- Tamaño de clusters:", np.bincount(clusters + 1 if -1 in clusters else clusters))  # Tamaño de cada cluster

# 8. Interpretación (simulando datos reales)
# Ejemplo de interpretación de los clusters
print("\nInterpretación (ejemplo segmentación clientes):")
print("Cluster 1: Clientes de bajo valor")
print("Cluster 2: Clientes promedio")
print("Cluster 3: Clientes premium")