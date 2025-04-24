# Conjuntos Difusos: Representar qué tanto una temperatura pertenece a distintas categorías

# Función para calcular el grado de pertenencia al conjunto "frío"
def frio(temp):
    # Si la temperatura es menor o igual a 10°C, pertenece completamente al conjunto "frío"
    if temp <= 10:
        return 1.0
    # Si la temperatura está entre 10°C y 20°C, el grado de pertenencia disminuye linealmente
    elif temp <= 20:
        return (20 - temp) / 10
    # Si la temperatura es mayor a 20°C, no pertenece al conjunto "frío"
    else:
        return 0.0

# Función para calcular el grado de pertenencia al conjunto "templado"
def templado(temp):
    # Si la temperatura está entre 15°C y 25°C, el grado de pertenencia aumenta linealmente
    if 15 <= temp <= 25:
        return (temp - 15) / 10
    # Si la temperatura está entre 25°C y 35°C, el grado de pertenencia disminuye linealmente
    elif 25 < temp <= 35:
        return (35 - temp) / 10
    # Fuera de este rango, no pertenece al conjunto "templado"
    else:
        return 0.0

# Función para calcular el grado de pertenencia al conjunto "caliente"
def caliente(temp):
    # Si la temperatura es menor o igual a 30°C, no pertenece al conjunto "caliente"
    if temp <= 30:
        return 0.0
    # Si la temperatura está entre 30°C y 40°C, el grado de pertenencia aumenta linealmente
    elif temp <= 40:
        return (temp - 30) / 10
    # Si la temperatura es mayor a 40°C, pertenece completamente al conjunto "caliente"
    else:
        return 1.0

# Probamos con una temperatura específica
temp = 27

# Imprimimos los resultados de los grados de pertenencia para cada conjunto difuso
print(f"Temperatura: {temp}°C")
print(f"Grado de FRÍO: {frio(temp):.2f}")       # Grado de pertenencia al conjunto "frío"
print(f"Grado de TEMPLADO: {templado(temp):.2f}") # Grado de pertenencia al conjunto "templado"
print(f"Grado de CALIENTE: {caliente(temp):.2f}") # Grado de pertenencia al conjunto "caliente"
