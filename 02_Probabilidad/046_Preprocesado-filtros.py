# --- Importación de librerías necesarias ---
import numpy as np  # Librería para operaciones matemáticas avanzadas, como manejo de arreglos y generación de números aleatorios
import matplotlib.pyplot as plt  # Librería para visualización de datos y gráficos
from scipy.ndimage import convolve  # Función para aplicar convoluciones a imágenes
from skimage import data, img_as_float  # Librería para trabajar con imágenes; incluye funciones y conjuntos de datos

# --- Definición de funciones para preprocesamiento de imágenes ---

def gaussian_kernel(size=3, sigma=1.0):
    """
    Genera un kernel gaussiano 2D (distribución normal).
    Este kernel se utiliza para suavizar imágenes, reduciendo el ruido.
    
    Parámetros:
    - size: Tamaño del kernel (debe ser impar para tener un centro definido).
    - sigma: Desviación estándar de la distribución gaussiana (controla la suavidad).
    
    Retorna:
    - kernel: Matriz 2D normalizada que representa el kernel gaussiano.
    """
    kernel = np.zeros((size, size))  # Inicializar el kernel como una matriz de ceros
    center = size // 2  # Calcular el índice del centro del kernel
    for i in range(size):  # Recorrer filas del kernel
        for j in range(size):  # Recorrer columnas del kernel
            x, y = i - center, j - center  # Coordenadas relativas al centro
            # Aplicar la fórmula de la distribución gaussiana
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))
    return kernel / np.sum(kernel)  # Normalizar el kernel para que la suma de sus valores sea 1

def bilateral_filter(image, sigma_spatial=1, sigma_intensity=0.1):
    """
    Aplica un filtro bilateral a la imagen.
    Este filtro combina proximidad espacial y similitud de intensidad para preservar bordes.
    
    Parámetros:
    - image: Imagen de entrada (2D, en escala de grises).
    - sigma_spatial: Desviación estándar para el kernel espacial (proximidad entre píxeles).
    - sigma_intensity: Desviación estándar para la similitud de intensidad (diferencia de valores de píxeles).
    
    Retorna:
    - result: Imagen filtrada.
    """
    result = np.zeros_like(image)  # Inicializar la imagen de salida con ceros (mismo tamaño que la entrada)
    size = int(3 * sigma_spatial) * 2 + 1  # Calcular el tamaño del kernel basado en sigma_spatial
    spatial_kernel = gaussian_kernel(size, sigma_spatial)  # Generar el kernel espacial gaussiano
    
    # Recorrer cada píxel de la imagen (excluyendo bordes para evitar errores de índice)
    for i in range(size // 2, image.shape[0] - size // 2):
        for j in range(size // 2, image.shape[1] - size // 2):
            # Extraer un parche (ventana) alrededor del píxel actual
            patch = image[i - size // 2:i + size // 2 + 1, j - size // 2:j + size // 2 + 1]
            # Calcular la similitud de intensidad entre el parche y el píxel central
            intensity_diff = np.exp(-(patch - image[i, j])**2 / (2 * sigma_intensity**2))
            # Combinar el kernel espacial con la similitud de intensidad
            combined_kernel = spatial_kernel * intensity_diff
            # Calcular el valor filtrado como promedio ponderado
            result[i, j] = np.sum(patch * combined_kernel) / np.sum(combined_kernel)
    return result  # Retornar la imagen filtrada

# --- Simulación y visualización ---

# Cargar una imagen de ejemplo desde la librería skimage y convertirla a formato de punto flotante
image = img_as_float(data.camera())  # Imagen en escala de grises (0 a 1)

# Añadir ruido gaussiano a la imagen para simular una imagen ruidosa
noisy_image = image + np.random.normal(0, 0.1, image.shape)  # Ruido con media 0 y desviación estándar 0.1

# Aplicar un filtro gaussiano para suavizar la imagen (reduce ruido pero puede difuminar bordes)
gaussian_filtered = convolve(noisy_image, gaussian_kernel(5, 1))  # Kernel de tamaño 5x5 y sigma=1

# Aplicar un filtro bilateral para preservar bordes mientras se reduce el ruido
bilateral_filtered = bilateral_filter(noisy_image, sigma_spatial=2, sigma_intensity=0.2)

# Mostrar los resultados en una figura con 3 subgráficos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Crear una figura con 1 fila y 3 columnas
titles = ['Imagen con Ruido', 'Filtro Gaussiano', 'Filtro Bilateral']  # Títulos de los subgráficos
images = [noisy_image, gaussian_filtered, bilateral_filtered]  # Lista de imágenes a mostrar

# Recorrer cada subgráfico y mostrar la imagen correspondiente
for ax, title, img in zip(axes, titles, images):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)  # Mostrar la imagen en escala de grises
    ax.set_title(title)  # Establecer el título del subgráfico
    ax.axis('off')  # Ocultar los ejes para una visualización más limpia

plt.tight_layout()  # Ajustar el diseño para evitar superposición entre subgráficos
plt.show()  # Mostrar la figura con las imágenes