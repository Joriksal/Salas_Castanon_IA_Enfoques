# --- Importación de librerías necesarias ---
import numpy as np  # Librería para operaciones matemáticas avanzadas y manejo de arreglos
from sklearn.naive_bayes import GaussianNB  # Modelo de clasificación Naive Bayes Gaussiano
from sklearn.model_selection import train_test_split  # Función para dividir los datos en entrenamiento y prueba
from skimage import data, color, exposure  # Funciones para trabajar con imágenes (carga, conversión y análisis)
import matplotlib.pyplot as plt  # Librería para visualización de datos y gráficos

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
    X = np.array([extract_features(img) for img in X_class0 + X_class1])  # Características extraídas
    y = np.array([0] * 10 + [1] * 10)  # Etiquetas: 0 para astronauta, 1 para café
    return X, y

# --- Paso 2: Entrenamiento del modelo ---
# Cargar el conjunto de datos
X, y = load_skimage_examples()

# Dividir los datos en entrenamiento (70%) y prueba (30%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar un modelo Naive Bayes Gaussiano
model = GaussianNB()  # Inicializamos el modelo
model.fit(X_train, y_train)  # Entrenamos el modelo con los datos de entrenamiento

# Evaluar la precisión del modelo en el conjunto de prueba
accuracy = model.score(X_test, y_test)  # Calculamos la precisión del modelo
print(f"Precisión del modelo: {accuracy * 100:.2f}%")  # Mostramos la precisión en porcentaje

# --- Paso 3: Predicción con una nueva imagen ---
def predict_example():
    """
    Realiza una predicción sobre una nueva imagen (en este caso, 'astronaut').
    """
    example_image = data.astronaut()  # Imagen de ejemplo (astronauta)
    hsv = color.rgb2hsv(example_image)  # Convertir la imagen de RGB a HSV
    hist = exposure.histogram(hsv[:, :, 0], nbins=16)[0]  # Calcular histograma del canal H (matiz)
    hist = hist / hist.sum()  # Normalizar el histograma
    pred = model.predict([hist])[0]  # Realizar la predicción con el modelo entrenado
    print(f"Predicción: {'Astronauta' if pred == 0 else 'Café'}")  # Mostrar el resultado de la predicción

# --- Paso 4: Visualización de imágenes de ejemplo ---
# Mostrar las imágenes de ejemplo utilizadas en el conjunto de datos
plt.figure(figsize=(10, 5))  # Crear una figura de tamaño 10x5 pulgadas
plt.subplot(121)  # Primer subgráfico (1 fila, 2 columnas, posición 1)
plt.imshow(data.astronaut())  # Mostrar la imagen del astronauta
plt.title("Astronauta (Clase 0)")  # Título del subgráfico
plt.subplot(122)  # Segundo subgráfico (posición 2)
plt.imshow(data.coffee())  # Mostrar la imagen del café
plt.title("Café (Clase 1)")  # Título del subgráfico
plt.show()  # Mostrar la figura con las imágenes

# --- Paso 5: Ejecutar predicción ---
predict_example()  # Llamamos a la función para realizar una predicción