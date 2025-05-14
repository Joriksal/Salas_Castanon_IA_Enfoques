import random  # Se utiliza para generar números aleatorios, lo cual es útil para inicializar estados aleatorios
               # y generar vecinos aleatorios en el espacio de búsqueda.

from copy import deepcopy  # Permite realizar copias profundas de objetos, asegurando que las modificaciones
                           # en las copias no afecten al objeto original. Es esencial para trabajar con estructuras
                           # complejas como listas o diccionarios en algoritmos como la búsqueda tabú.

class BusquedaTabu:
    def __init__(self, funcion_objetivo, generar_vecino, tamano_tabu=10, max_iter=100):
        """
        Inicializa el algoritmo de búsqueda tabú.
        
        Args:
            funcion_objetivo: Función a minimizar.
            generar_vecino: Función que genera los vecinos de un estado.
            tamano_tabu: Tamaño máximo de la lista tabú.
            max_iter: Número máximo de iteraciones permitidas.
        """
        self.funcion = funcion_objetivo  # Función objetivo que se desea minimizar.
        self.generar_vecino = generar_vecino  # Función para generar vecinos de un estado.
        self.tamano_tabu = tamano_tabu  # Tamaño máximo de la lista tabú.
        self.max_iter = max_iter  # Número máximo de iteraciones.

    def resolver(self, estado_inicial):
        """
        Ejecuta el algoritmo de búsqueda tabú.
        
        Args:
            estado_inicial: Estado inicial desde donde comienza la búsqueda.
        
        Returns:
            tuple: (mejor_global, mejor_valor, historial)
                - mejor_global: El mejor estado encontrado.
                - mejor_valor: El valor de la función objetivo en el mejor estado.
                - historial: Lista de diccionarios con el progreso de cada iteración.
        """
        # Inicialización
        mejor_global = deepcopy(estado_inicial)  # Copia profunda del estado inicial como mejor estado global.
        mejor_valor = self.funcion(mejor_global)  # Evaluar la función objetivo en el mejor estado global.
        actual = deepcopy(estado_inicial)  # Copia profunda del estado inicial como estado actual.
        lista_tabu = []  # Lista tabú para evitar ciclos (almacena estados ya visitados).
        historial = []  # Historial para registrar el progreso de cada iteración.

        for iteracion in range(self.max_iter):  # Bucle principal que itera hasta el máximo permitido.
            # Generar vecinos que no estén en la lista tabú.
            vecinos = [v for v in self.generar_vecino(actual) if v not in lista_tabu]
            
            if not vecinos:  # Si no hay vecinos válidos (todos están en la lista tabú), detener la búsqueda.
                break

            # Evaluar los vecinos para encontrar el mejor.
            vecino_actual = None  # Inicializar el mejor vecino como None.
            mejor_valor_vecino = float('inf')  # Inicializar el mejor valor vecino con infinito (para minimizar).

            for vecino in vecinos:  # Iterar sobre los vecinos generados.
                valor = self.funcion(vecino)  # Evaluar la función objetivo en el vecino.
                if valor < mejor_valor_vecino:  # Si el vecino es mejor, actualizar.
                    mejor_valor_vecino = valor
                    vecino_actual = vecino

            # Actualizar el mejor estado global si se encuentra una mejora.
            if mejor_valor_vecino < mejor_valor:
                mejor_global = deepcopy(vecino_actual)  # Actualizar el mejor estado global.
                mejor_valor = mejor_valor_vecino  # Actualizar el mejor valor global.

            # Mover al mejor vecino (aunque no mejore el estado actual).
            actual = vecino_actual
            
            # Actualizar la lista tabú (FIFO: First In, First Out).
            lista_tabu.append(deepcopy(actual))  # Agregar el estado actual a la lista tabú.
            if len(lista_tabu) > self.tamano_tabu:  # Si la lista tabú excede su tamaño, eliminar el más antiguo.
                lista_tabu.pop(0)

            # Registrar el progreso en el historial.
            historial.append({
                'iteracion': iteracion,  # Número de iteración.
                'estado': deepcopy(actual),  # Estado actual.
                'valor': mejor_valor_vecino,  # Valor del mejor vecino.
                'mejor_global': mejor_valor  # Mejor valor global encontrado hasta ahora.
            })

        return mejor_global, mejor_valor, historial  # Retornar el mejor estado, valor y el historial.

# ------------------------------------------
# EJEMPLOS DE USO
# ------------------------------------------

# Ejemplo 1: Minimizar una función matemática.
def funcion_ejemplo(x):
    """
    Función matemática a minimizar: x^4 - 3x^3 + 2x^2 - x + 5.
    """
    return x**4 - 3*x**3 + 2*x**2 - x + 5

def vecindario_continuo(x, paso=0.1, n_vecinos=5):
    """
    Genera vecinos de un estado en un espacio continuo.
    
    Args:
        x: Estado actual.
        paso: Rango de variación para generar vecinos.
        n_vecinos: Número de vecinos a generar.
    
    Returns:
        list: Lista de estados vecinos.
    """
    return [x + random.uniform(-paso, paso) for _ in range(n_vecinos)]  # Generar vecinos aleatorios.

# Ejemplo 2: Problema de la Mochila.
def funcion_mochila(solucion, pesos, valores, capacidad):
    """
    Calcula el valor total de una solución para el problema de la mochila.
    
    Args:
        solucion: Lista binaria que indica qué objetos están en la mochila.
        pesos: Lista de pesos de los objetos.
        valores: Lista de valores de los objetos.
        capacidad: Capacidad máxima de la mochila.
    
    Returns:
        float: Valor total de la solución (negativo si excede la capacidad).
    """
    peso_total = sum(p * s for p, s in zip(pesos, solucion))  # Calcular el peso total de la solución.
    if peso_total > capacidad:  # Si el peso excede la capacidad, la solución es inválida.
        return float('inf')  # Retornar infinito para indicar una solución inválida.
    return -sum(v * s for v, s in zip(valores, solucion))  # Retornar el valor total (negativo para minimizar).

def generar_vecino_mochila(solucion):
    """
    Genera vecinos de una solución para el problema de la mochila.
    
    Args:
        solucion: Lista binaria que indica qué objetos están en la mochila.
    
    Returns:
        list: Lista de soluciones vecinas.
    """
    vecinos = []
    for i in range(len(solucion)):  # Iterar sobre cada elemento de la solución.
        vecino = solucion[:]  # Copiar la solución actual.
        vecino[i] = 1 - vecino[i]  # Cambiar 0 a 1 o 1 a 0.
        vecinos.append(vecino)  # Agregar el vecino generado.
    return vecinos

if __name__ == "__main__":
    # Ejemplo 1: Minimización de una función matemática.
    print("=== EJEMPLO 1: MINIMIZACIÓN FUNCIÓN MATEMÁTICA ===")
    tabu = BusquedaTabu(
        funcion_ejemplo,  # Función objetivo.
        lambda x: vecindario_continuo(x, 0.5, 10),  # Función para generar vecinos.
        tamano_tabu=5,  # Tamaño de la lista tabú.
        max_iter=50  # Número máximo de iteraciones.
    )
    
    x_inicial = random.uniform(-2, 5)  # Generar un estado inicial aleatorio.
    mejor_x, mejor_valor, hist = tabu.resolver(x_inicial)
    
    print(f"Solución inicial: x = {x_inicial:.4f}, f(x) = {funcion_ejemplo(x_inicial):.4f}")
    print(f"Solución encontrada: x = {mejor_x:.4f}, f(x) = {mejor_valor:.4f}")
    print(f"Mejoría: {100*(funcion_ejemplo(x_inicial)-mejor_valor)/funcion_ejemplo(x_inicial):.2f}%")
    
    # Ejemplo 2: Problema de la Mochila.
    print("\n=== EJEMPLO 2: PROBLEMA DE LA MOCHILA ===")
    pesos = [2, 3, 5, 7, 9]  # Pesos de los objetos.
    valores = [10, 15, 25, 40, 60]  # Valores de los objetos.
    capacidad = 15  # Capacidad máxima de la mochila.
    
    tabu_mochila = BusquedaTabu(
        lambda s: funcion_mochila(s, pesos, valores, capacidad),  # Función objetivo.
        generar_vecino_mochila,  # Función para generar vecinos.
        tamano_tabu=3,  # Tamaño de la lista tabú.
        max_iter=100  # Número máximo de iteraciones.
    )
    
    sol_inicial = [random.randint(0, 1) for _ in valores]  # Generar una solución inicial aleatoria.
    mejor_sol, mejor_val, _ = tabu_mochila.resolver(sol_inicial)
    
    print(f"Solución inicial: {sol_inicial}")
    print(f"Valor inicial: {-funcion_mochila(sol_inicial, pesos, valores, capacidad)}")
    print(f"\nMejor solución: {mejor_sol}")
    print(f"Peso total: {sum(p*m for p,m in zip(pesos, mejor_sol))}")
    print(f"Valor total: {-mejor_val}")