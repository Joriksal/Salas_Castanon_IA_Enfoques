# Importamos la librería `random` para generar números aleatorios.
# Esta librería será utilizada para crear perturbaciones aleatorias en los estados vecinos.
import random  

def hill_climbing(funcion_objetivo, estado_inicial, generar_vecino, max_iter=1000):
    """
    Implementación del algoritmo de Hill Climbing para maximizar una función objetivo.
    Este algoritmo busca encontrar el máximo local de una función mediante iteraciones.

    Args:
        funcion_objetivo (function): Función que se desea maximizar. Debe ser una función que tome un estado como entrada y devuelva un valor numérico.
                                     Ejemplo: lambda x: -x**2 + 4*x.
        estado_inicial: El punto de inicio para la búsqueda. Puede ser cualquier valor válido para la función objetivo.
                        Ejemplo: 0.0.
        generar_vecino (function): Función que genera un estado vecino a partir del estado actual. 
                                   Ejemplo: lambda x: x + random.uniform(-0.1, 0.1).
        max_iter (int): Número máximo de iteraciones permitidas para el algoritmo. Por defecto, es 1000.

    Returns:
        tuple: Una tupla que contiene:
            - mejor_estado: El estado que maximiza la función objetivo.
            - mejor_valor: El valor de la función objetivo en el mejor estado encontrado.
    """
    # Inicializamos el estado actual con el estado inicial proporcionado.
    estado_actual = estado_inicial
    # Inicializamos el mejor estado como el estado inicial.
    mejor_estado = estado_actual
    # Calculamos el valor de la función objetivo en el estado inicial.
    mejor_valor = funcion_objetivo(estado_actual)

    # Iteramos hasta alcanzar el número máximo de iteraciones permitido.
    for _ in range(max_iter):
        # Generamos un estado vecino a partir del estado actual utilizando la función `generar_vecino`.
        vecino = generar_vecino(estado_actual)
        # Calculamos el valor de la función objetivo en el estado vecino generado.
        valor_vecino = funcion_objetivo(vecino)
        
        # Si el valor del estado vecino es mejor que el valor actual:
        if valor_vecino > mejor_valor:
            # Actualizamos el mejor estado y el mejor valor con los del vecino.
            mejor_estado, mejor_valor = vecino, valor_vecino
            # Nos movemos al estado vecino, es decir, actualizamos el estado actual.
            estado_actual = vecino
        else:
            # Si el vecino no mejora el valor actual, asumimos que hemos alcanzado un óptimo local.
            # Terminamos el bucle antes de completar todas las iteraciones.
            break
    
    # Devolvemos el mejor estado y el mejor valor encontrados durante la ejecución del algoritmo.
    return (mejor_estado, mejor_valor)

# Bloque principal del programa: Este bloque se ejecuta solo si el archivo se ejecuta directamente.
if __name__ == "__main__":
    # Definimos la función objetivo que queremos maximizar.
    # En este caso, es una parábola invertida: f(x) = -x^2 + 4x.
    # El máximo de esta función está en x = 2.
    funcion = lambda x: -x**2 + 4*x

    # Definimos el estado inicial desde donde comenzará la búsqueda.
    # En este caso, comenzamos en x = 0.0.
    inicio = 0.0

    # Definimos cómo generar estados vecinos a partir del estado actual.
    # Utilizamos una perturbación aleatoria en el rango [-0.5, 0.5].
    generar_vecino = lambda x: x + random.uniform(-0.5, 0.5)
    
    # Ejecutamos el algoritmo Hill Climbing con los parámetros definidos.
    mejor_estado, mejor_valor = hill_climbing(funcion, inicio, generar_vecino)

    # Mostramos el resultado final en la consola.
    # Esto incluye el mejor estado encontrado (x) y el valor de la función objetivo en ese estado (f(x)).
    print(f"Máximo encontrado en x = {mejor_estado:.2f}, f(x) = {mejor_valor:.2f}")
    # Salida esperada: "Máximo encontrado en x = 2.00, f(x) = 4.00"