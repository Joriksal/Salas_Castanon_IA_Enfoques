import numpy as np
import matplotlib.pyplot as plt
from skimage import data, color, feature
from sklearn.cluster import MeanShift, estimate_bandwidth
from scipy import ndimage

# --- 1. Preprocesado (Filtro Gaussiano para reducir ruido) ---
# Cargamos una imagen de ejemplo (astronauta) y la convertimos a escala de grises
image = color.rgb2gray(data.astronaut())
# Aplicamos un filtro Gaussiano para suavizar la imagen y reducir el ruido
image = ndimage.gaussian_filter(image, sigma=1)

# --- 2. Detección de Bordes (Canny con umbrales automáticos) ---
# Usamos el detector de bordes Canny para identificar los bordes en la imagen suavizada
# El parámetro sigma controla la suavidad del gradiente antes de detectar bordes
edges = feature.canny(image, sigma=2, low_threshold=None, high_threshold=None)

# --- 3. Segmentación Probabilística (Mean-Shift) ---
def segment_mean_shift(image_rgb, quantile=0.1):
    """
    Segmentación basada en densidad de color y posición usando Mean-Shift.
    Args:
        image_rgb: Imagen en formato RGB.
        quantile: Cuantil para estimar el ancho de banda (bandwidth) del algoritmo.
    Returns:
        labels: Matriz con las etiquetas de los segmentos.
    """
    # Aplanamos la imagen en un arreglo 2D donde cada fila es un píxel (R, G, B)
    flat_image = np.reshape(image_rgb, [-1, 3])
    # Estimamos el ancho de banda para el algoritmo Mean-Shift
    bandwidth = estimate_bandwidth(flat_image, quantile=quantile, n_samples=500)
    # Creamos el modelo Mean-Shift con el ancho de banda estimado
    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
    # Ajustamos el modelo a los datos de la imagen
    ms.fit(flat_image)
    # Obtenemos las etiquetas de los segmentos
    labels = ms.labels_
    # Reestructuramos las etiquetas en la forma original de la imagen
    return np.reshape(labels, image_rgb.shape[:2])

# Aplicamos la segmentación Mean-Shift a la imagen original en color
image_rgb = data.astronaut()  # Imagen original en RGB
segmented = segment_mean_shift(image_rgb)

# --- Visualización ---
# Creamos una figura con 3 subgráficos para mostrar los resultados
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
titles = ['Imagen Original', 'Bordes (Canny)', 'Segmentación (Mean-Shift)']
# Preparamos las imágenes a mostrar: original, bordes y segmentación
images = [image_rgb, edges, color.label2rgb(segmented, image_rgb, kind='avg')]

# Iteramos sobre los subgráficos para mostrar cada imagen con su título
for ax, title, img in zip(axes, titles, images):
    ax.imshow(img)  # Mostramos la imagen
    ax.set_title(title)  # Asignamos el título
    ax.axis('off')  # Ocultamos los ejes

# Ajustamos el diseño de la figura y mostramos los resultados
plt.tight_layout()
plt.show()