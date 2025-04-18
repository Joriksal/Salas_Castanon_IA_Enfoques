import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage import data

# 1. Cargar imagen en escala de grises
# Usamos una imagen de ejemplo proporcionada por skimage (imagen de una cámara)
gray = data.camera()
gray = cv2.resize(gray, (320, 240))  # Redimensionamos la imagen a 320x240 píxeles

# 2. Calcular histograma y probabilidad
# Calculamos el histograma de la imagen en escala de grises
hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
# Normalizamos el histograma para obtener la probabilidad de cada nivel de intensidad
prob = hist / np.sum(hist)

# 3. Crear máscara de sombra
# Creamos un mapa de probabilidades para cada píxel basado en su intensidad
prob_map = prob[gray].squeeze()  # Eliminamos dimensiones extra para simplificar

# Definimos un umbral para identificar sombras:
# - Pixeles con alta probabilidad (frecuentes en la imagen)
# - Pixeles oscuros (intensidad menor a 100)
umbral_sombra = 0.015
sombra_mask = np.where((prob_map > umbral_sombra) & (gray < 100), 255, 0).astype(np.uint8)

# 4. Detección de texturas usando Sobel
# Calculamos los gradientes en las direcciones X e Y usando el operador Sobel
sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)  # Gradiente en X
sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)  # Gradiente en Y
# Calculamos la magnitud del gradiente para identificar texturas
textura = cv2.magnitude(sobelx, sobely)
# Convertimos la magnitud a un formato de 8 bits para visualización
textura = np.uint8(textura)

# 5. Mostrar resultados
# Creamos una figura con 3 subgráficos para mostrar los resultados
plt.figure(figsize=(12, 6))

# Subgráfico 1: Imagen original
plt.subplot(1, 3, 1)
plt.title('Imagen Original')
plt.imshow(gray, cmap='gray')  # Mostramos la imagen en escala de grises
plt.axis('off')  # Ocultamos los ejes

# Subgráfico 2: Máscara de sombra
plt.subplot(1, 3, 2)
plt.title('Máscara de Sombra')
plt.imshow(sombra_mask, cmap='gray')  # Mostramos la máscara de sombra
plt.axis('off')  # Ocultamos los ejes

# Subgráfico 3: Zonas con textura
plt.subplot(1, 3, 3)
plt.title('Zonas con Textura')
plt.imshow(textura, cmap='gray')  # Mostramos las zonas con textura
plt.axis('off')  # Ocultamos los ejes

# Ajustamos el diseño para evitar superposición y mostramos la figura
plt.tight_layout()
plt.show()
