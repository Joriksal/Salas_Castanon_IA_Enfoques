import random  # Importamos la librería para generar números aleatorios

def hill_climbing(funcion_objetivo, estado_inicial, generar_vecino, max_iter=1000):
    """
    Implementación del algoritmo de Hill Climbing para maximizar una función objetivo.

    Args:
        funcion_objetivo (function): Función a maximizar (ejemplo: lambda x: -x**2 + 4*x).
        estado_inicial: Punto de inicio para la búsqueda (ejemplo: 0.0).
        generar_vecino (function): Función que genera un estado vecino 
                                   (ejemplo: lambda x: x + random.uniform(-0.1, 0.1)).
        max_iter (int): Número máximo de iteraciones permitidas.

    Returns:
        Tupla (mejor_estado, mejor_valor): El mejor estado encontrado y su valor asociado.
    """
    # Inicializamos el estado actual y el mejor estado con el estado inicial
    estado_actual = estado_inicial
    mejor_estado = estado_actual
    mejor_valor = funcion_objetivo(estado_actual)  # Evaluamos el valor inicial de la función objetivo

    # Iteramos hasta el máximo de iteraciones permitido
    for _ in range(max_iter):
        # Generamos un estado vecino a partir del estado actual
        vecino = generar_vecino(estado_actual)
        # Calculamos el valor de la función objetivo en el estado vecino
        valor_vecino = funcion_objetivo(vecino)
        
        # Si el vecino mejora el valor actual, actualizamos el mejor estado y valor
        if valor_vecino > mejor_valor:
            mejor_estado, mejor_valor = vecino, valor_vecino
            estado_actual = vecino  # Nos movemos al vecino mejor
        else:
            # Si no hay mejora, asumimos que hemos alcanzado un óptimo local y terminamos
            break
    
    # Devolvemos el mejor estado y valor encontrados
    return (mejor_estado, mejor_valor)

# Ejemplo de uso: Maximizar la función f(x) = -x^2 + 4x, cuyo máximo está en x=2
if __name__ == "__main__":
    # Definimos la función objetivo
    funcion = lambda x: -x**2 + 4*x
    # Definimos el estado inicial
    inicio = 0.0
    # Definimos cómo generar estados vecinos (perturbación aleatoria)
    generar_vecino = lambda x: x + random.uniform(-0.5, 0.5)
    
    # Ejecutamos el algoritmo Hill Climbing
    mejor_estado, mejor_valor = hill_climbing(funcion, inicio, generar_vecino)
    
    # Mostramos el resultado
    print(f"Máximo encontrado en x = {mejor_estado:.2f}, f(x) = {mejor_valor:.2f}")
    # Salida esperada: "Máximo encontrado en x = 2.00, f(x) = 4.00"