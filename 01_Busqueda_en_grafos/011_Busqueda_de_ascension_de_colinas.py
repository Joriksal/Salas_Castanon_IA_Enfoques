import random  # Para generar números aleatorios y mezclar listas.
from math import sqrt  # Para calcular raíces cuadradas (usado en la distancia euclidiana).

class HillClimbing:
    def __init__(self, funcion_objetivo, vecindario, max_iter=1000):
        """
        Inicializa el algoritmo de ascensión de colinas.
        
        Args:
            funcion_objetivo: Función a maximizar o minimizar.
            vecindario: Función que genera los vecinos de un estado.
            max_iter: Número máximo de iteraciones permitidas.
        """
        self.funcion = funcion_objetivo  # Almacena la función objetivo.
        self.vecindario = vecindario  # Almacena la función que genera vecinos.
        self.max_iter = max_iter  # Define el número máximo de iteraciones.

    def resolver(self, estado_inicial):
        """
        Ejecuta el algoritmo de ascensión de colinas.
        
        Args:
            estado_inicial: Estado inicial desde donde comienza la búsqueda.
        
        Returns:
            tuple: (mejor_estado, mejor_valor, historial)
                - mejor_estado: El mejor estado encontrado.
                - mejor_valor: El valor de la función objetivo en el mejor estado.
                - historial: Lista de (estado, valor) en cada iteración.
        """
        actual = estado_inicial  # Estado actual de la búsqueda.
        valor_actual = self.funcion(actual)  # Evalúa el estado inicial con la función objetivo.
        historial = [(actual, valor_actual)]  # Historial para registrar el progreso.

        for _ in range(self.max_iter):  # Iterar hasta el máximo permitido.
            vecinos = self.vecindario(actual)  # Generar los vecinos del estado actual.
            if not vecinos:  # Si no hay vecinos, detener la búsqueda.
                break

            # Evaluar todos los vecinos para encontrar el mejor.
            mejor_vecino = None
            mejor_valor = -float('inf')  # Inicializar con un valor muy bajo.

            for vecino in vecinos:
                valor = self.funcion(vecino)  # Evaluar la función objetivo en el vecino.
                if valor > mejor_valor:  # Si el vecino es mejor, actualizar.
                    mejor_valor = valor
                    mejor_vecino = vecino

            # Criterio de parada: Si no hay mejora, detener la búsqueda.
            if mejor_valor <= valor_actual:
                break

            # Actualizar el estado actual al mejor vecino.
            actual = mejor_vecino
            valor_actual = mejor_valor
            historial.append((actual, valor_actual))  # Registrar el progreso.

        return actual, valor_actual, historial  # Retornar el mejor estado, valor y el historial.

# ------------------------------------------
# EJEMPLOS DE USO
# ------------------------------------------

# Ejemplo 1: Maximizar una función matemática.
def funcion_ejemplo(x):
    """
    Función matemática a maximizar: -x^2 + 4x.
    Tiene un máximo en x=2 con valor f(x)=4.
    """
    return -x**2 + 4*x  # Calcula el valor de la función para un valor dado de x.

def vecindario_simple(x, paso=0.1):
    """
    Genera vecinos de un estado en un espacio unidimensional.
    
    Args:
        x: Estado actual.
        paso: Incremento o decremento para generar vecinos.
    
    Returns:
        list: Lista de estados vecinos.
    """
    return [x - paso, x + paso]  # Genera dos vecinos: uno a la izquierda y otro a la derecha.

# Ejemplo 2: Problema del Viajante (TSP simplificado).
def distancia(ciudad1, ciudad2):
    """
    Calcula la distancia euclidiana entre dos ciudades.
    
    Args:
        ciudad1: Coordenadas (x, y) de la primera ciudad.
        ciudad2: Coordenadas (x, y) de la segunda ciudad.
    
    Returns:
        float: Distancia entre las dos ciudades.
    """
    return sqrt((ciudad1[0] - ciudad2[0])**2 + (ciudad1[1] - ciudad2[1])**2)  # Fórmula de distancia euclidiana.

def funcion_tsp(ruta, ciudades):
    """
    Calcula la distancia total de una ruta en el problema del viajante.
    
    Args:
        ruta: Lista de índices que representan el orden de las ciudades.
        ciudades: Diccionario con las coordenadas de las ciudades.
    
    Returns:
        float: Distancia total de la ruta (negativa para maximizar).
    """
    return -sum(distancia(ciudades[ruta[i]], ciudades[ruta[i+1]]) 
               for i in range(len(ruta) - 1))  # Suma las distancias entre ciudades consecutivas.

def vecindario_tsp(ruta):
    """
    Genera vecinos de una ruta intercambiando dos ciudades.
    
    Args:
        ruta: Lista de índices que representan el orden de las ciudades.
    
    Returns:
        list: Lista de rutas vecinas.
    """
    vecinos = []
    for i in range(1, len(ruta) - 1):  # Evitar intercambiar la primera y última ciudad.
        for j in range(i + 1, len(ruta)):
            nueva_ruta = ruta[:]
            nueva_ruta[i], nueva_ruta[j] = nueva_ruta[j], nueva_ruta[i]  # Intercambiar dos ciudades.
            vecinos.append(nueva_ruta)  # Agregar la nueva ruta a la lista de vecinos.
    return vecinos

if __name__ == "__main__":
    # Ejemplo 1: Maximización de una función matemática.
    print("=== EJEMPLO 1: MAXIMIZACIÓN FUNCIÓN MATEMÁTICA ===")
    hc = HillClimbing(funcion_ejemplo, lambda x: vecindario_simple(x, 0.01))  # Crear instancia de HillClimbing.
    mejor_x, mejor_valor, historial = hc.resolver(random.uniform(-10, 10))  # Resolver con un estado inicial aleatorio.
    
    print(f"Solución encontrada: x = {mejor_x:.4f}, f(x) = {mejor_valor:.4f}")
    print(f"Pasos: {len(historial)}")  # Mostrar el número de pasos realizados.
    
    # Ejemplo 2: Problema del Viajante (TSP).
    print("\n=== EJEMPLO 2: PROBLEMA DEL VIAJANTE (TSP) ===")
    ciudades = {
        0: (0, 0),
        1: (1, 5),
        2: (3, 2),
        3: (5, 6),
        4: (8, 3)
    }
    
    # Generar una ruta inicial aleatoria.
    ruta_inicial = list(ciudades.keys())
    random.shuffle(ruta_inicial)  # Mezclar las ciudades para obtener una ruta inicial aleatoria.
    
    # Resolver el problema del viajante con Hill Climbing.
    hc_tsp = HillClimbing(
        lambda r: funcion_tsp(r, ciudades),  # Función objetivo para el TSP.
        vecindario_tsp,  # Función que genera vecinos.
        max_iter=100  # Número máximo de iteraciones.
    )
    
    mejor_ruta, mejor_dist, hist = hc_tsp.resolver(ruta_inicial)  # Resolver el TSP.
    print(f"Mejor ruta encontrada: {mejor_ruta}")
    print(f"Distancia total: {-mejor_dist:.2f}")  # Negativo porque la función objetivo devuelve valores negativos.
    print(f"Evaluaciones: {len(hist)}")  # Mostrar el número de evaluaciones realizadas.