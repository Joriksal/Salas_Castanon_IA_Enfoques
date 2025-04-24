import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt

# Definición del universo de discurso (rango de temperaturas en °C)
# Este rango va de 0 a 40 con incrementos de 1
x_temp = np.arange(0, 41, 1)

# Definición de los conjuntos difusos para las categorías de temperatura
# 'frio', 'templado' y 'caliente' se definen mediante funciones de membresía triangulares (trimf)

# Conjunto difuso "frío": 
# La membresía es máxima (1) entre 0 y 20, y disminuye linealmente hacia 0 fuera de este rango
frio = fuzz.trimf(x_temp, [0, 0, 20])

# Conjunto difuso "templado": 
# La membresía es máxima (1) en 25, y disminuye linealmente hacia 0 en los extremos (15 y 35)
templado = fuzz.trimf(x_temp, [15, 25, 35])

# Conjunto difuso "caliente": 
# La membresía es máxima (1) entre 30 y 40, y disminuye linealmente hacia 0 fuera de este rango
caliente = fuzz.trimf(x_temp, [30, 40, 40])

# Visualización de los conjuntos difusos
# Se genera una gráfica para mostrar las funciones de membresía de cada conjunto

plt.figure()  # Crear una nueva figura para la gráfica
plt.plot(x_temp, frio, 'b', label='Frío')  # Graficar el conjunto "frío" en azul
plt.plot(x_temp, templado, 'g', label='Templado')  # Graficar el conjunto "templado" en verde
plt.plot(x_temp, caliente, 'r', label='Caliente')  # Graficar el conjunto "caliente" en rojo

# Configuración de la gráfica
plt.title('Conjuntos Difusos de Temperatura')  # Título de la gráfica
plt.xlabel('Temperatura (°C)')  # Etiqueta del eje X
plt.ylabel('Grado de membresía')  # Etiqueta del eje Y
plt.legend()  # Mostrar la leyenda para identificar los conjuntos
plt.grid(True)  # Mostrar una cuadrícula para facilitar la lectura
plt.show()  # Mostrar la gráfica
