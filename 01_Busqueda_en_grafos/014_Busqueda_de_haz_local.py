import heapq  # Librería para trabajar con colas de prioridad (usada en heapq.nlargest)
import random  # Librería para generar números aleatorios (usada en varios lugares)

class LocalBeamSearch:
    def __init__(self, funcion_objetivo, generar_sucesores, k=3, max_iter=100):
        """
        Inicializa el algoritmo de Búsqueda de Haz Local.
        
        Args:
            funcion_objetivo: Función a maximizar.
            generar_sucesores: Función que genera los sucesores de un estado.
            k: Número de estados a mantener en el haz.
            max_iter: Número máximo de iteraciones.
        """
        self.funcion = funcion_objetivo  # Función objetivo que se busca maximizar.
        self.generar_sucesores = generar_sucesores  # Función para generar estados sucesores.
        self.k = k  # Número de estados en el haz.
        self.max_iter = max_iter  # Número máximo de iteraciones.

    def resolver(self, estados_iniciales):
        """
        Ejecuta la búsqueda de haz local.
        
        Args:
            estados_iniciales: Lista de k estados iniciales.
        
        Returns:
            tuple: (mejor_estado, mejor_valor, historial)
                - mejor_estado: El mejor estado encontrado.
                - mejor_valor: El valor de la función objetivo en el mejor estado.
                - historial: Lista de datos históricos de cada iteración.
        """
        # Verificar que tenemos al menos k estados iniciales.
        if len(estados_iniciales) < self.k:
            # Si no hay suficientes estados iniciales, se rellenan con copias aleatorias.
            estados_iniciales += [random.choice(estados_iniciales) for _ in range(self.k - len(estados_iniciales))]
        
        # Inicializar los k estados actuales.
        estados_actuales = estados_iniciales[:self.k]
        mejor_estado = None  # Variable para almacenar el mejor estado encontrado.
        mejor_valor = -float('inf')  # Inicializar con un valor muy bajo.
        historial = []  # Historial para registrar el progreso.

        for iteracion in range(self.max_iter):  # Bucle principal de iteraciones.
            # Generar todos los sucesores de los estados actuales.
            todos_sucesores = []
            for estado in estados_actuales:
                sucesores = self.generar_sucesores(estado)  # Generar sucesores del estado actual.
                todos_sucesores.extend(sucesores)  # Agregar los sucesores a la lista total.
            
            # Evaluar todos los sucesores.
            sucesores_evaluados = []
            for sucesor in todos_sucesores:
                valor = self.funcion(sucesor)  # Evaluar la función objetivo en cada sucesor.
                sucesores_evaluados.append((valor, sucesor))  # Guardar el valor y el sucesor.
            
            # Seleccionar los k mejores sucesores usando una cola de prioridad.
            mejores = heapq.nlargest(self.k, sucesores_evaluados, key=lambda x: x[0])
            
            # Actualizar el mejor estado global si se encuentra uno mejor.
            mejor_actual_val, mejor_actual_estado = mejores[0]
            if mejor_actual_val > mejor_valor:
                mejor_valor = mejor_actual_val
                mejor_estado = mejor_actual_estado
            
            # Preparar los estados actuales para la siguiente iteración.
            estados_actuales = [estado for (val, estado) in mejores]
            
            # Registrar el progreso en el historial.
            historial.append({
                'iteracion': iteracion,
                'mejor_valor': mejor_valor,
                'estados_actuales': [estado for (val, estado) in mejores],
                'valores_actuales': [val for (val, estado) in mejores]
            })
            
            # Condición de parada: Si todos los valores son iguales al mejor valor.
            if all(val == mejor_valor for val, _ in mejores):
                break
        
        return mejor_estado, mejor_valor, historial

# ------------------------------------------
# EJEMPLOS DE USO
# ------------------------------------------

# Ejemplo 1: Maximizar función matemática
def funcion_ejemplo(x):
    """
    Función matemática a maximizar: -x^4 + 3x^3 - 2x^2 + x + 5.
    Tiene múltiples máximos locales.
    """
    return -x**4 + 3*x**3 - 2*x**2 + x + 5

def generar_sucesores_continuo(x, paso=0.1, n_sucesores=3):
    """
    Genera sucesores en un espacio continuo mediante pequeñas perturbaciones.
    
    Args:
        x: Estado actual.
        paso: Rango de variación para generar sucesores.
        n_sucesores: Número de sucesores a generar.
    
    Returns:
        list: Lista de estados sucesores.
    """
    return [x + random.uniform(-paso, paso) for _ in range(n_sucesores)]

# Ejemplo 2: Problema de las N-Reinas
def conflicto_reinas(tablero):
    """
    Calcula el número de conflictos en un tablero de N-Reinas.
    Los conflictos incluyen ataques horizontales, verticales y diagonales.
    
    Args:
        tablero: Lista donde el índice representa la fila y el valor la columna de la reina.
    
    Returns:
        int: Número de conflictos (a minimizar).
    """
    n = len(tablero)
    conflictos = 0
    for i in range(n):
        for j in range(i+1, n):
            # Conflictos horizontales y diagonales.
            if tablero[i] == tablero[j] or abs(tablero[i] - tablero[j]) == abs(i - j):
                conflictos += 1
    return conflictos

def generar_sucesores_reinas(tablero):
    """
    Genera sucesores moviendo una reina a una posición diferente en su columna.
    
    Args:
        tablero: Lista donde el índice representa la fila y el valor la columna de la reina.
    
    Returns:
        list: Lista de tableros sucesores.
    """
    sucesores = []
    n = len(tablero)
    for i in range(n):
        for j in range(n):
            if j != tablero[i]:  # Evitar la posición actual.
                nuevo_tablero = list(tablero)
                nuevo_tablero[i] = j
                sucesores.append(tuple(nuevo_tablero))
    return sucesores

if __name__ == "__main__":
    # Ejemplo 1: Maximización de una función matemática.
    print("=== EJEMPLO 1: MAXIMIZACIÓN FUNCIÓN MATEMÁTICA ===")
    lbs = LocalBeamSearch(
        funcion_ejemplo,
        lambda x: generar_sucesores_continuo(x, 0.5, 5),
        k=3,
        max_iter=50
    )
    
    # Generar k estados iniciales aleatorios.
    estados_iniciales = [random.uniform(-2, 5) for _ in range(3)]
    mejor_x, mejor_valor, hist = lbs.resolver(estados_iniciales)
    
    print(f"Estados iniciales: {[round(x, 2) for x in estados_iniciales]}")
    print(f"Mejor solución: x = {round(mejor_x, 4)}, f(x) = {round(mejor_valor, 4)}")
    
    # Ejemplo 2: Problema de las N-Reinas.
    print("\n=== EJEMPLO 2: PROBLEMA DE LAS N-REINAS (8x8) ===")
    n = 8
    lbs_reinas = LocalBeamSearch(
        lambda t: -conflicto_reinas(t),  # Convertimos a problema de maximización.
        generar_sucesores_reinas,
        k=5,
        max_iter=100
    )
    
    # Generar k estados iniciales aleatorios.
    estados_iniciales = [tuple(random.randint(0, n-1) for _ in range(n)) for _ in range(5)]
    mejor_tablero, mejor_conflictos, _ = lbs_reinas.resolver(estados_iniciales)
    
    print(f"Mejor tablero encontrado: {mejor_tablero}")
    print(f"Número de conflictos: {-mejor_conflictos}")
    print("Visualización:")
    for fila in range(n):
        linea = ['Q' if col == mejor_tablero[fila] else '.' for col in range(n)]
        print(' '.join(linea))