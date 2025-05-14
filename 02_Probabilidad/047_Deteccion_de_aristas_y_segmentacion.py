# --- Importación de librerías necesarias ---
import numpy as np  # Librería para operaciones matemáticas avanzadas y manejo de arreglos
import matplotlib.pyplot as plt  # Librería para visualización de datos y gráficos
from skimage import data, color, feature  # Funciones para trabajar con imágenes (carga, conversión y detección de bordes)
from sklearn.cluster import MeanShift, estimate_bandwidth  # Algoritmo de clustering Mean-Shift y estimación de ancho de banda
from scipy import ndimage  # Librería para procesamiento de imágenes, como filtros y transformaciones

# --- 1. Preprocesado (Filtro Gaussiano para reducir ruido) ---
# Cargamos una imagen de ejemplo (astronauta) y la convertimos a escala de grises
image = color.rgb2gray(data.astronaut())  # Convertimos la imagen RGB a escala de grises
# Aplicamos un filtro Gaussiano para suavizar la imagen y reducir el ruido
# El parámetro sigma controla la intensidad del suavizado
image = ndimage.gaussian_filter(image, sigma=1)  # Suavizado con sigma=1

# --- 2. Detección de Bordes (Canny con umbrales automáticos) ---
# Usamos el detector de bordes Canny para identificar los bordes en la imagen suavizada
# El parámetro sigma controla la suavidad del gradiente antes de detectar bordes
edges = feature.canny(image, sigma=2, low_threshold=None, high_threshold=None)  # Detección de bordes con sigma=2

# --- 3. Segmentación Probabilística (Mean-Shift) ---
def segment_mean_shift(image_rgb, quantile=0.1):
    """
    Segmentación basada en densidad de color y posición usando Mean-Shift.
    Este método agrupa píxeles similares en color y posición espacial.
    
    Args:
        image_rgb: Imagen en formato RGB.
        quantile: Cuantil para estimar el ancho de banda (bandwidth) del algoritmo.
    
    Returns:
        labels: Matriz con las etiquetas de los segmentos.
    """
    # Aplanamos la imagen en un arreglo 2D donde cada fila es un píxel (R, G, B)
    flat_image = np.reshape(image_rgb, [-1, 3])  # Convertimos la imagen 3D (alto, ancho, canales) a 2D (píxeles, canales)
    # Estimamos el ancho de banda para el algoritmo Mean-Shift
    bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=500)  # Estimación basada en una muestra de píxeles
    # Creamos el modelo Mean-Shift con el ancho de banda estimado
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)  # Inicializamos el modelo con bin_seeding para acelerar el cálculo
    # Ajustamos el modelo a los datos de la imagen
    ms.fit(flat_image)  # Agrupamos los píxeles en segmentos
    # Obtenemos las etiquetas de los segmentos
    labels = ms.labels_  # Cada píxel recibe una etiqueta que indica a qué segmento pertenece
    # Reestructuramos las etiquetas en la forma original de la imagen
    return np.reshape(labels, image_rgb.shape[:2])  # Convertimos las etiquetas a la forma (alto, ancho)

# Aplicamos la segmentación Mean-Shift a la imagen original en color
image_rgb = data.astronaut()  # Cargamos la imagen original en formato RGB
segmented = segment_mean_shift(image_rgb)  # Aplicamos el algoritmo de segmentación

# --- Visualización ---
# Creamos una figura con 3 subgráficos para mostrar los resultados
fig, axes = plt.subplots(1, 3, figsize=(18, 6))  # Figura con 1 fila y 3 columnas
titles = ['Imagen Original', 'Bordes (Canny)', 'Segmentación (Mean-Shift)']  # Títulos de los subgráficos
# Preparamos las imágenes a mostrar: original, bordes y segmentación
images = [
    image_rgb,  # Imagen original en color
    edges,  # Imagen con bordes detectados
    color.label2rgb(segmented, image_rgb, kind='avg')  # Imagen segmentada con colores promedio por segmento
]

# Iteramos sobre los subgráficos para mostrar cada imagen con su título
for ax, title, img in zip(axes, titles, images):
    ax.imshow(img)  # Mostramos la imagen en el subgráfico
    ax.set_title(title)  # Asignamos el título correspondiente
    ax.axis('off')  # Ocultamos los ejes para una visualización más limpia

# Ajustamos el diseño de la figura y mostramos los resultados
plt.tight_layout()  # Ajustamos el diseño para evitar superposición entre subgráficos
plt.show()  # Mostramos la figura con las imágenes