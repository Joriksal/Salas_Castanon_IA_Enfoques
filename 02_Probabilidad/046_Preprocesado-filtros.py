import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import convolve
from skimage import data, img_as_float

def gaussian_kernel(size=3, sigma=1.0):
    """
    Genera un kernel gaussiano 2D (distribución normal).
    Este kernel se utiliza para suavizar imágenes, reduciendo el ruido.
    
    Parámetros:
    - size: Tamaño del kernel (debe ser impar).
    - sigma: Desviación estándar de la distribución gaussiana.
    
    Retorna:
    - kernel: Matriz 2D normalizada que representa el kernel gaussiano.
    """
    kernel = np.zeros((size, size))
    center = size // 2  # Centro del kernel
    for i in range(size):
        for j in range(size):
            x, y = i - center, j - center  # Coordenadas relativas al centro
            kernel[i, j] = np.exp(-(x**2 + y**2) / (2 * sigma**2))  # Fórmula gaussiana
    return kernel / np.sum(kernel)  # Normalizar para que la suma sea 1

def bilateral_filter(image, sigma_spatial=1, sigma_intensity=0.1):
    """
    Aplica un filtro bilateral a la imagen.
    Este filtro combina proximidad espacial y similitud de intensidad para preservar bordes.
    
    Parámetros:
    - image: Imagen de entrada (2D).
    - sigma_spatial: Desviación estándar para el kernel espacial (proximidad).
    - sigma_intensity: Desviación estándar para la similitud de intensidad.
    
    Retorna:
    - result: Imagen filtrada.
    """
    result = np.zeros_like(image)  # Inicializar la imagen de salida
    size = int(3 * sigma_spatial) * 2 + 1  # Tamaño del kernel basado en sigma_spatial
    spatial_kernel = gaussian_kernel(size, sigma_spatial)  # Kernel espacial gaussiano
    
    # Recorrer cada píxel de la imagen (excluyendo bordes)
    for i in range(size//2, image.shape[0] - size//2):
        for j in range(size//2, image.shape[1] - size//2):
            # Extraer un parche alrededor del píxel actual
            patch = image[i-size//2:i+size//2+1, j-size//2:j+size//2+1]
            # Calcular la similitud de intensidad entre el parche y el píxel central
            intensity_diff = np.exp(-(patch - image[i, j])**2 / (2 * sigma_intensity**2))
            # Combinar el kernel espacial con la similitud de intensidad
            combined_kernel = spatial_kernel * intensity_diff
            # Calcular el valor filtrado como promedio ponderado
            result[i, j] = np.sum(patch * combined_kernel) / np.sum(combined_kernel)
    return result

# --- Simulación y visualización ---
# Cargar imagen de ejemplo y convertirla a formato de punto flotante
image = img_as_float(data.camera())

# Añadir ruido gaussiano a la imagen
noisy_image = image + np.random.normal(0, 0.1, image.shape)  # Ruido con media 0 y desviación estándar 0.1

# Aplicar filtro gaussiano para suavizar la imagen
gaussian_filtered = convolve(noisy_image, gaussian_kernel(5, 1))  # Kernel de tamaño 5x5 y sigma=1

# Aplicar filtro bilateral para preservar bordes mientras se reduce el ruido
bilateral_filtered = bilateral_filter(noisy_image, sigma_spatial=2, sigma_intensity=0.2)

# Mostrar resultados en una figura con 3 subgráficos
fig, axes = plt.subplots(1, 3, figsize=(15, 5))  # Crear una figura con 1 fila y 3 columnas
titles = ['Imagen con Ruido', 'Filtro Gaussiano', 'Filtro Bilateral']  # Títulos de los subgráficos
images = [noisy_image, gaussian_filtered, bilateral_filtered]  # Imágenes a mostrar

# Recorrer cada subgráfico y mostrar la imagen correspondiente
for ax, title, img in zip(axes, titles, images):
    ax.imshow(img, cmap='gray', vmin=0, vmax=1)  # Mostrar imagen en escala de grises
    ax.set_title(title)  # Establecer título
    ax.axis('off')  # Ocultar ejes

plt.tight_layout()  # Ajustar diseño para evitar superposición
plt.show()  # Mostrar la figura