# --- Importación de librerías necesarias ---
import cv2  # Librería para procesamiento de imágenes y visión por computadora
import numpy as np  # Librería para operaciones matemáticas avanzadas y manejo de arreglos
import matplotlib.pyplot as plt  # Librería para visualización de datos y gráficos
from skimage import data  # Librería para trabajar con imágenes; incluye conjuntos de datos de ejemplo

# --- 1. Cargar imagen en escala de grises ---
# Usamos una imagen de ejemplo proporcionada por skimage (imagen de una cámara)
gray = data.camera()  # Cargamos la imagen en escala de grises (0 a 255)
gray = cv2.resize(gray, (320, 240))  # Redimensionamos la imagen a 320x240 píxeles para reducir el tamaño

# --- 2. Calcular histograma y probabilidad ---
# Calculamos el histograma de la imagen en escala de grises
# cv2.calcHist calcula la frecuencia de cada nivel de intensidad (0-255)
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])

# Normalizamos el histograma para obtener la probabilidad de cada nivel de intensidad
# Dividimos cada frecuencia entre la suma total de frecuencias
prob = hist / np.sum(hist)

# --- 3. Crear máscara de sombra ---
# Creamos un mapa de probabilidades para cada píxel basado en su intensidad
# Usamos el histograma normalizado para asignar una probabilidad a cada nivel de intensidad
prob_map = prob[gray].squeeze()  # Eliminamos dimensiones extra para simplificar el arreglo

# Definimos un umbral para identificar sombras:
# - Pixeles con alta probabilidad (frecuentes en la imagen)
# - Pixeles oscuros (intensidad menor a 100)
umbral_sombra = 0.015  # Umbral de probabilidad para considerar un píxel como sombra
sombra_mask = np.where((prob_map > umbral_sombra) & (gray < 100), 255, 0).astype(np.uint8)
# np.where genera una máscara binaria: 255 para sombras, 0 para el resto

# --- 4. Detección de texturas usando Sobel ---
# Calculamos los gradientes en las direcciones X e Y usando el operador Sobel
# Sobel detecta cambios de intensidad en la imagen, útiles para identificar bordes y texturas
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente en la dirección X
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente en la dirección Y

# Calculamos la magnitud del gradiente para identificar texturas
# La magnitud combina los gradientes en X e Y para obtener la intensidad total del cambio
textura = cv2.magnitude(sobelx, sobely)

# Convertimos la magnitud a un formato de 8 bits para visualización
# Esto permite mostrar la imagen de texturas en escala de grises
textura = np.uint8(textura)

# --- 5. Mostrar resultados ---
# Creamos una figura con 3 subgráficos para mostrar los resultados
plt.figure(figsize=(12, 6))  # Tamaño de la figura: 12x6 pulgadas

# Subgráfico 1: Imagen original
plt.subplot(1, 3, 1)  # Primer subgráfico (1 fila, 3 columnas, posición 1)
plt.title('Imagen Original')  # Título del subgráfico
plt.imshow(gray, cmap='gray')  # Mostramos la imagen en escala de grises
plt.axis('off')  # Ocultamos los ejes para una visualización más limpia

# Subgráfico 2: Máscara de sombra
plt.subplot(1, 3, 2)  # Segundo subgráfico (posición 2)
plt.title('Máscara de Sombra')  # Título del subgráfico
plt.imshow(sombra_mask, cmap='gray')  # Mostramos la máscara de sombra
plt.axis('off')  # Ocultamos los ejes para una visualización más limpia

# Subgráfico 3: Zonas con textura
plt.subplot(1, 3, 3)  # Tercer subgráfico (posición 3)
plt.title('Zonas con Textura')  # Título del subgráfico
plt.imshow(textura, cmap='gray')  # Mostramos las zonas con textura
plt.axis('off')  # Ocultamos los ejes para una visualización más limpia

# Ajustamos el diseño para evitar superposición entre subgráficos
plt.tight_layout()  # Ajustamos automáticamente los márgenes y el espaciado
plt.show()  # Mostramos la figura con los resultados
