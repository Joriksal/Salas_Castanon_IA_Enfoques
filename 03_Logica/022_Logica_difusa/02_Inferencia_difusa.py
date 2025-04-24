# Inferencia Difusa simple: Activar ventilador según temperatura
# Este programa utiliza lógica difusa para calcular la velocidad del ventilador
# en función de la temperatura proporcionada.

def velocidad_ventilador(temp):
    """
    Calcula la velocidad del ventilador basada en la temperatura usando lógica difusa.

    Parámetros:
    temp (float): Temperatura en grados Celsius.

    Retorna:
    float: Velocidad sugerida del ventilador en porcentaje.
    """
    # Definimos las funciones de pertenencia para las categorías de temperatura:
    # - Frío: Máxima pertenencia a 20°C, disminuye linealmente hasta 10°C o menos.
    frio = max(0, min(1, (20 - temp) / 10))
    
    # - Templado: Máxima pertenencia entre 25°C y 15°C, disminuye linealmente hacia 35°C o menos de 15°C.
    templado = max(0, min(1, (temp - 15) / 10 if temp < 25 else (35 - temp) / 10))
    
    # - Caliente: Máxima pertenencia a 30°C, aumenta linealmente desde 30°C en adelante.
    caliente = max(0, min(1, (temp - 30) / 10))

    # Inferencia difusa: Combinamos las reglas para calcular la velocidad del ventilador.
    # Reglas:
    # - Si la temperatura es fría, el ventilador debe ir lento (30%).
    # - Si la temperatura es templada, el ventilador debe ir a velocidad media (60%).
    # - Si la temperatura es caliente, el ventilador debe ir rápido (100%).
    velocidad = (frio * 30 + templado * 60 + caliente * 100) / (frio + templado + caliente)
    
    return velocidad

# Probamos el sistema con una temperatura de ejemplo
temp = 28  # Temperatura en grados Celsius
print(f"\nTemperatura: {temp}°C")
print(f"Velocidad sugerida del ventilador: {velocidad_ventilador(temp):.2f}%")
