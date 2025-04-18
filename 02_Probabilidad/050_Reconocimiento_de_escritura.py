import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# 1. Cargar dataset MNIST (dígitos 0-9)
# El dataset MNIST contiene imágenes de dígitos escritos a mano (28x28 píxeles).
# Cada imagen está representada como un vector de 784 características (28x28 = 784 píxeles).
mnist = fetch_openml('mnist_784', version=1, as_frame=False)
X, y = mnist.data, mnist.target  # X contiene las imágenes, y contiene las etiquetas (dígitos 0-9).

# 2. Preprocesamiento: Binarizar imágenes
# Convertimos los valores de los píxeles en binarios (0 o 1).
# Esto es útil porque el modelo BernoulliNB funciona mejor con datos binarios.
X_binary = np.where(X > 0, 1, 0)  # Los píxeles > 0 se convierten en 1, los demás en 0.

# 3. Dividir datos en entrenamiento y prueba
# Dividimos el dataset en un conjunto de entrenamiento (80%) y un conjunto de prueba (20%).
X_train, X_test, y_train, y_test = train_test_split(X_binary, y, test_size=0.2, random_state=42)

# 4. Entrenar modelo Naive Bayes
# Usamos el clasificador BernoulliNB, que es una variante de Naive Bayes diseñada para datos binarios.
model = BernoulliNB()
model.fit(X_train, y_train)  # Entrenamos el modelo con los datos de entrenamiento.

# 5. Evaluar precisión
# Predecimos las etiquetas para el conjunto de prueba y calculamos la precisión del modelo.
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)  # Calculamos la precisión comparando predicciones con etiquetas reales.
print(f"Precisión del modelo: {accuracy * 100:.2f}%")  # Mostramos la precisión en porcentaje.

# 6. Mostrar una predicción ejemplo
# Seleccionamos una imagen del conjunto de prueba para visualizarla y predecir su etiqueta.
sample_index = 8  # Índice de la imagen de prueba que queremos mostrar.
sample_image = X_test[sample_index].reshape(28, 28)  # Reshapeamos el vector a una matriz 28x28.
predicted_digit = model.predict([X_test[sample_index]])[0]  # Predecimos el dígito para esta imagen.

# Mostramos la imagen y la predicción del modelo.
plt.imshow(sample_image, cmap='binary')  # Mostramos la imagen en escala de grises (binaria).
plt.title(f"Predicción: {predicted_digit}")  # Título con la predicción del modelo.
plt.show()