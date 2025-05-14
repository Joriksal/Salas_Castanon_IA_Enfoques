import random  # Importa el módulo random para la generación de números aleatorios.
import math    # Importa el módulo math para funciones matemáticas.

class SimulatedAnnealing:
    def __init__(self, funcion_objetivo, generar_vecino, temp_inicial=1000, 
                 temp_final=0.1, enfriamiento=0.95, iter_por_temp=100):
        """
        Inicializa el algoritmo de Temple Simulado.
        
        Args:
            funcion_objetivo: Función a minimizar.
            generar_vecino: Función que genera un vecino aleatorio.
            temp_inicial: Temperatura inicial (controla la exploración).
            temp_final: Temperatura final (criterio de parada).
            enfriamiento: Factor de reducción de temperatura (0 < enfriamiento < 1).
            iter_por_temp: Número de iteraciones por cada temperatura.
        """
        # Almacena los parámetros del algoritmo.
        self.funcion = funcion_objetivo  # Función objetivo a minimizar.
        self.generar_vecino = generar_vecino  # Función para generar vecinos.
        self.temp_inicial = temp_inicial  # Temperatura inicial.
        self.temp_final = temp_final  # Temperatura final (criterio de parada).
        self.enfriamiento = enfriamiento  # Factor de enfriamiento.
        self.iter_por_temp = iter_por_temp  # Iteraciones por temperatura.

    def resolver(self, estado_inicial):
        """
        Ejecuta el algoritmo de Temple Simulado.
        
        Args:
            estado_inicial: Estado inicial desde donde comienza la búsqueda.
        
        Returns:
            tuple: (mejor_estado, mejor_valor, historial)
                - mejor_estado: El mejor estado encontrado.
                - mejor_valor: El valor de la función objetivo en el mejor estado.
                - historial: Lista de datos históricos de cada iteración.
        """
        # Inicialización de variables.
        actual = estado_inicial  # Estado actual.
        valor_actual = self.funcion(actual)  # Valor de la función objetivo en el estado actual.
        mejor_estado = actual  # Mejor estado encontrado globalmente.
        mejor_valor = valor_actual  # Mejor valor de la función objetivo.
        temperatura = self.temp_inicial  # Temperatura inicial.
        historial = []  # Historial para registrar el progreso.

        # Bucle principal: continúa mientras la temperatura sea mayor que la final.
        while temperatura > self.temp_final:
            for _ in range(self.iter_por_temp):  # Iterar un número fijo de veces por temperatura.
                vecino = self.generar_vecino(actual)  # Generar un vecino aleatorio.
                valor_vecino = self.funcion(vecino)  # Evaluar la función objetivo en el vecino.

                # Diferencia de energía (queremos minimizar).
                delta = valor_vecino - valor_actual

                # Criterio de aceptación:
                # Aceptar si el vecino es mejor o con una probabilidad basada en la temperatura.
                if delta < 0 or random.random() < math.exp(-delta / temperatura):
                    actual = vecino  # Actualizar el estado actual.
                    valor_actual = valor_vecino  # Actualizar el valor actual.

                    # Actualizar el mejor estado global si el vecino es mejor.
                    if valor_actual < mejor_valor:
                        mejor_estado = actual
                        mejor_valor = valor_actual

            # Registrar datos históricos.
            historial.append({
                'temperatura': temperatura,  # Temperatura actual.
                'mejor_valor': mejor_valor,  # Mejor valor encontrado hasta ahora.
                'valor_actual': valor_actual  # Valor actual en esta iteración.
            })

            # Reducir la temperatura (enfriamiento).
            temperatura *= self.enfriamiento

        # Retornar el mejor estado, valor y el historial.
        return mejor_estado, mejor_valor, historial

# ------------------------------------------
# EJEMPLOS DE USO
# ------------------------------------------

# Ejemplo 1: Minimizar función matemática
def funcion_ejemplo(x):
    """
    Función matemática a minimizar: x^4 - 3x^3 + 2x^2 - x + 5.
    Tiene múltiples mínimos locales.
    """
    return x**4 - 3*x**3 + 2*x**2 - x + 5

def vecindario_continuo(x):
    """
    Genera un vecino en un espacio continuo mediante una pequeña perturbación aleatoria.
    
    Args:
        x: Estado actual.
    
    Returns:
        float: Nuevo estado vecino.
    """
    return x + random.uniform(-0.5, 0.5)

# Ejemplo 2: Problema del Viajante (TSP)
def distancia(p1, p2):
    """
    Calcula la distancia euclidiana entre dos puntos.
    
    Args:
        p1: Coordenadas del primer punto (x, y).
        p2: Coordenadas del segundo punto (x, y).
    
    Returns:
        float: Distancia euclidiana entre los puntos.
    """
    return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

def funcion_tsp(ruta, ciudades):
    """
    Calcula la distancia total de una ruta en el problema del viajante.
    
    Args:
        ruta: Lista de índices que representan el orden de las ciudades.
        ciudades: Diccionario con las coordenadas de las ciudades.
    
    Returns:
        float: Distancia total de la ruta.
    """
    return sum(distancia(ciudades[ruta[i]], ciudades[ruta[(i+1) % len(ruta)]]) 
               for i in range(len(ruta)))

def generar_vecino_tsp(ruta):
    """
    Genera un vecino de la ruta intercambiando dos ciudades aleatorias.
    
    Args:
        ruta: Lista de índices que representan el orden de las ciudades.
    
    Returns:
        list: Nueva ruta vecina.
    """
    i, j = random.sample(range(len(ruta)), 2)
    vecino = ruta[:]
    vecino[i], vecino[j] = vecino[j], vecino[i]
    return vecino

if __name__ == "__main__":
    # Ejemplo 1: Minimización de una función matemática.
    print("=== EJEMPLO 1: MINIMIZACIÓN FUNCIÓN MATEMÁTICA ===")
    sa = SimulatedAnnealing(
        funcion_ejemplo,
        vecindario_continuo,
        temp_inicial=1000,
        temp_final=0.01,
        enfriamiento=0.95,
        iter_por_temp=100
    )
    
    x_inicial = random.uniform(-2, 5)  # Generar un estado inicial aleatorio.
    mejor_x, mejor_valor, hist = sa.resolver(x_inicial)
    
    print(f"Solución inicial: x = {x_inicial:.4f}, f(x) = {funcion_ejemplo(x_inicial):.4f}")
    print(f"Solución encontrada: x = {mejor_x:.4f}, f(x) = {mejor_valor:.4f}")
    print(f"Mejoría: {100*(funcion_ejemplo(x_inicial)-mejor_valor)/abs(funcion_ejemplo(x_inicial)):.2f}%")

    # Ejemplo 2: Problema del Viajante (TSP).
    print("\n=== EJEMPLO 2: PROBLEMA DEL VIAJANTE (TSP) ===")
    # Coordenadas de ciudades.
    ciudades = {
        0: (0, 0),
        1: (1, 5),
        2: (3, 2),
        3: (5, 6),
        4: (8, 3)
    }
    
    # Generar una ruta inicial aleatoria.
    ruta_inicial = list(ciudades.keys())
    random.shuffle(ruta_inicial)
    
    sa_tsp = SimulatedAnnealing(
        lambda r: funcion_tsp(r, ciudades),
        generar_vecino_tsp,
        temp_inicial=10000,
        temp_final=0.1,
        enfriamiento=0.99,
        iter_por_temp=100
    )
    
    mejor_ruta, mejor_dist, _ = sa_tsp.resolver(ruta_inicial)
    print(f"Ruta inicial: {ruta_inicial}")
    print(f"Distancia inicial: {funcion_tsp(ruta_inicial, ciudades):.2f}")
    print(f"\nMejor ruta encontrada: {mejor_ruta}")
    print(f"Distancia total: {mejor_dist:.2f}")