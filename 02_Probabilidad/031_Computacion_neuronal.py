import numpy as np  # Para realizar operaciones matemáticas y manejo de arreglos de manera eficiente.
import matplotlib.pyplot as plt  # Para graficar datos y visualizar la frontera de decisión.

# Datos de entrenamiento: Compuerta AND (x1, x2, etiqueta)
# Cada fila de X representa una entrada (x1, x2) y y contiene la salida esperada.
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y = np.array([0, 0, 0, 1])  # Salida esperada para la compuerta AND

# Inicialización de pesos y sesgo (valores aleatorios pequeños para comenzar)
weights = np.random.randn(2) * 0.1  # Pesos iniciales para x1 y x2
bias = np.random.randn() * 0.1       # Sesgo inicial

# Hiperparámetros
learning_rate = 0.1  # Tasa de aprendizaje para ajustar los pesos
epochs = 10          # Número de iteraciones sobre el conjunto de datos

# Función de activación (Step function para perceptrón)
# Devuelve 1 si la entrada es mayor o igual a 0, de lo contrario devuelve 0.
def step_function(x):
    return 1 if x >= 0 else 0

# Entrenamiento del perceptrón utilizando la Regla de Hebb modificada
for epoch in range(epochs):  # Iterar sobre el número de épocas
    for i in range(len(X)):  # Iterar sobre cada muestra de entrenamiento
        # Paso 1: Calcular la salida de la neurona (producto punto + sesgo)
        z = np.dot(X[i], weights) + bias  # z = w1*x1 + w2*x2 + b
        prediction = step_function(z)    # Aplicar la función de activación
        
        # Paso 2: Calcular el error y actualizar pesos y sesgo
        error = y[i] - prediction  # Diferencia entre la salida esperada y la predicción
        weights += learning_rate * error * X[i]  # Actualizar pesos
        bias += learning_rate * error            # Actualizar sesgo
        
    # Imprimir los pesos y el sesgo después de cada época
    print(f"Epoch {epoch + 1}, Pesos: {weights}, Sesgo: {bias:.2f}")

# Predicción final después del entrenamiento
print("\nPruebas finales:")
for i in range(len(X)):
    z = np.dot(X[i], weights) + bias  # Calcular z para cada entrada
    print(f"Entrada: {X[i]}, Predicción: {step_function(z)}")  # Mostrar predicción

# Visualización de los datos y la frontera de decisión
plt.figure(figsize=(8, 5))  # Configurar el tamaño de la figura

# Graficar los puntos de datos
for i, point in enumerate(X):
    # Usar color rojo para la clase 1 y azul para la clase 0
    plt.scatter(point[0], point[1], color='red' if y[i] == 1 else 'blue', s=100)
    
# Graficar la línea de decisión (w1*x1 + w2*x2 + b = 0)
x_line = np.linspace(-0.5, 1.5, 100)  # Valores de x para la línea
y_line = (-weights[0] * x_line - bias) / weights[1]  # Calcular y para la línea
plt.plot(x_line, y_line, 'k--', label='Frontera de decisión')  # Línea discontinua negra

# Configurar etiquetas y título del gráfico
plt.xlabel('x1')  # Etiqueta del eje x
plt.ylabel('x2')  # Etiqueta del eje y
plt.title('Perceptrón para Compuerta AND')  # Título del gráfico
plt.legend()  # Mostrar leyenda
plt.grid()  # Mostrar cuadrícula
plt.show()  # Mostrar el gráfico