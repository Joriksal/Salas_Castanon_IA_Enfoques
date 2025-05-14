# Importación de librerías necesarias
import numpy as np  # Librería para manejo de arreglos y operaciones matemáticas
import matplotlib.pyplot as plt  # Librería para generar gráficos
from sklearn.cluster import KMeans  # Algoritmo de agrupamiento K-Means
from sklearn.datasets import make_blobs  # Generador de datos sintéticos para pruebas
from sklearn.metrics import silhouette_score  # Métrica para evaluar la calidad del agrupamiento
from sklearn.preprocessing import StandardScaler  # Herramienta para normalizar datos

# 1. Generación de datos sintéticos (3 grupos claros + ruido)
np.random.seed(42)  # Fijamos la semilla para garantizar reproducibilidad de resultados
# Generamos un conjunto de datos con 3 clusters bien definidos
X, y_true = make_blobs(n_samples=300, centers=3, cluster_std=1.0, random_state=42)
# Añadimos ruido al conjunto de datos (20 puntos aleatorios distribuidos uniformemente)
X = np.vstack([X, np.random.uniform(low=-10, high=10, size=(20, 2))])

# 2. Preprocesamiento (normalización)
# Normalizamos los datos para que todas las características tengan media 0 y desviación estándar 1
scaler = StandardScaler()  # Inicializamos el escalador
X_scaled = scaler.fit_transform(X)  # Ajustamos y transformamos los datos

# 3. Método del codo para determinar k óptimo
# Inicializamos listas para almacenar métricas de evaluación
wcss = []  # Within-Cluster Sum of Squares (inercia): mide la compacidad de los clusters
silhouettes = []  # Coeficiente Silhouette: mide la separación entre clusters
k_range = range(2, 8)  # Rango de valores de k (número de clusters) a evaluar

# Iteramos sobre diferentes valores de k para encontrar el óptimo
for k in k_range:
    # Inicializamos el modelo K-Means con k clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    # Ajustamos el modelo a los datos normalizados
    kmeans.fit(X_scaled)
    # Calculamos y almacenamos la inercia (suma de distancias al centroide)
    wcss.append(kmeans.inertia_)
    # Calculamos y almacenamos el coeficiente Silhouette
    silhouettes.append(silhouette_score(X_scaled, kmeans.labels_))

# 4. Gráfica doble (Codo + Silhouette)
plt.figure(figsize=(12, 5))  # Configuramos el tamaño de la figura

# Gráfica del método del codo
plt.subplot(1, 2, 1)  # Primera subgráfica (1 fila, 2 columnas, posición 1)
plt.plot(k_range, wcss, 'bo-')  # Gráfica de inercia vs número de clusters
plt.xlabel('Número de clusters (k)')  # Etiqueta del eje x
plt.ylabel('WCSS (Inercia)')  # Etiqueta del eje y
plt.title('Método del Codo')  # Título de la gráfica

# Gráfica del coeficiente Silhouette
plt.subplot(1, 2, 2)  # Segunda subgráfica (1 fila, 2 columnas, posición 2)
plt.plot(k_range, silhouettes, 'go-')  # Gráfica de Silhouette vs número de clusters
plt.xlabel('Número de clusters (k)')  # Etiqueta del eje x
plt.ylabel('Coeficiente Silhouette')  # Etiqueta del eje y
plt.title('Análisis Silhouette')  # Título de la gráfica

plt.tight_layout()  # Ajustamos el diseño para evitar superposición
plt.show()  # Mostramos las gráficas

# 5. Agrupamiento final con k óptimo (k=3 según gráficas)
optimal_k = 3  # Seleccionamos el número óptimo de clusters basado en las gráficas
# Inicializamos el modelo K-Means con el número óptimo de clusters
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
# Ajustamos el modelo a los datos y obtenemos las etiquetas de los clusters
clusters = kmeans.fit_predict(X_scaled)

# 6. Visualización profesional de resultados
plt.figure(figsize=(10, 6))  # Configuramos el tamaño de la figura
# Definimos colores para los clusters
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A']

# Graficamos los puntos de cada cluster
for i in range(optimal_k):
    plt.scatter(X_scaled[clusters == i, 0], X_scaled[clusters == i, 1], 
                s=50, c=colors[i], label=f'Cluster {i+1}')

# Graficamos los centroides de los clusters
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1],
            s=200, marker='*', c='black', label='Centroides')

# Graficamos puntos etiquetados como ruido (si existieran)
if -1 in clusters:  # Verificamos si hay puntos etiquetados como ruido
    plt.scatter(X_scaled[clusters == -1, 0], X_scaled[clusters == -1, 1],
                s=50, c='gray', label='Ruido')

# Configuramos el título y etiquetas de los ejes
plt.title('Agrupamiento de Datos con K-Means\n(k={})'.format(optimal_k), fontsize=14)
plt.xlabel('Característica 1 (normalizada)', fontsize=12)
plt.ylabel('Característica 2 (normalizada)', fontsize=12)
plt.legend()  # Añadimos una leyenda para identificar clusters y centroides
plt.grid(True, alpha=0.3)  # Añadimos una cuadrícula con transparencia
plt.show()  # Mostramos la gráfica

# 7. Métricas de evaluación
# Mostramos métricas clave para evaluar el modelo
print(f"\nMétricas para k={optimal_k}:")
print(f"- Inercia: {kmeans.inertia_:.2f}")  # Inercia final del modelo
print(f"- Silhouette Score: {silhouette_score(X_scaled, clusters):.2f}")  # Coeficiente Silhouette
# Mostramos el tamaño de cada cluster
print("- Tamaño de clusters:", np.bincount(clusters + 1 if -1 in clusters else clusters))

# 8. Interpretación (simulando datos reales)
# Ejemplo de interpretación de los clusters (en un contexto de segmentación de clientes)
print("\nInterpretación (ejemplo segmentación clientes):")
print("Cluster 1: Clientes de bajo valor")
print("Cluster 2: Clientes promedio")
print("Cluster 3: Clientes premium")