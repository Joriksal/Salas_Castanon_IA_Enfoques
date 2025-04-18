import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import train_test_split
from skimage import data, color, exposure
import matplotlib.pyplot as plt

# --- Paso 1: Cargar imágenes a color disponibles en skimage ---
def load_skimage_examples():
    """
    Carga imágenes de ejemplo de la biblioteca skimage y genera un conjunto de datos
    con características extraídas del histograma del canal de matiz (Hue) en el espacio HSV.
    """
    # Usaremos 'astronaut' (clase 0) y 'coffee' (clase 1) - ambas son imágenes RGB
    img_class0 = data.astronaut()  # Imagen de un astronauta
    img_class1 = data.coffee()     # Imagen de una taza de café
    
    # Generar múltiples muestras añadiendo ruido gaussiano para simular variaciones
    np.random.seed(42)  # Fijar la semilla para reproducibilidad
    X_class0 = [np.clip(img_class0 + np.random.normal(0, 10, img_class0.shape), 0, 255) for _ in range(10)]
    X_class1 = [np.clip(img_class1 + np.random.normal(0, 10, img_class1.shape), 0, 255) for _ in range(10)]
    
    # Función para extraer características del histograma del canal de matiz (Hue)
    def extract_features(img):
        hsv = color.rgb2hsv(img)  # Convertir la imagen de RGB a HSV
        hist = exposure.histogram(hsv[:, :, 0], nbins=16)[0]  # Calcular histograma del canal H (matiz)
        return hist / hist.sum()  # Normalizar el histograma para que la suma sea 1
    
    # Extraer características de todas las imágenes y crear etiquetas
    X = np.array([extract_features(img) for img in X_class0 + X_class1])
    y = np.array([0] * 10 + [1] * 10)  # Etiquetas: 0 para astronauta, 1 para café
    return X, y

# --- Paso 2: Entrenamiento del modelo ---
# Cargar el conjunto de datos
X, y = load_skimage_examples()

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar un modelo Naive Bayes Gaussiano
model = GaussianNB()
model.fit(X_train, y_train)

# Evaluar la precisión del modelo en el conjunto de prueba
accuracy = model.score(X_test, y_test)
print(f"Precisión del modelo: {accuracy * 100:.2f}%")

# --- Paso 3: Predicción con una nueva imagen ---
def predict_example():
    """
    Realiza una predicción sobre una nueva imagen (en este caso, 'astronaut').
    """
    example_image = data.astronaut()  # Imagen de ejemplo
    hsv = color.rgb2hsv(example_image)  # Convertir a HSV
    hist = exposure.histogram(hsv[:, :, 0], nbins=16)[0]  # Calcular histograma del canal H
    hist = hist / hist.sum()  # Normalizar el histograma
    pred = model.predict([hist])[0]  # Realizar la predicción
    print(f"Predicción: {'Astronauta' if pred == 0 else 'Café'}")

# --- Paso 4: Visualización de imágenes de ejemplo ---
# Mostrar las imágenes de ejemplo utilizadas en el conjunto de datos
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(data.astronaut())  # Mostrar la imagen del astronauta
plt.title("Astronauta (Clase 0)")
plt.subplot(122)
plt.imshow(data.coffee())  # Mostrar la imagen del café
plt.title("Café (Clase 1)")
plt.show()

# --- Paso 5: Ejecutar predicción ---
predict_example()