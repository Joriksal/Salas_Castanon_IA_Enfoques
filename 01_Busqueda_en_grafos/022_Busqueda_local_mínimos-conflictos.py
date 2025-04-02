import random
from collections import defaultdict

class MinimosConflictos:
    def __init__(self, variables, dominios, restricciones, max_iter=10000):
        """
        Inicializa el solver de mínimos conflictos.
        
        Args:
            variables: Lista de variables del problema.
            dominios: Diccionario {variable: [valores posibles]}.
            restricciones: Diccionario {(var1, var2): función_restricción}.
            max_iter: Máximo número de iteraciones permitidas.
        """
        self.variables = variables
        self.dominios = dominios
        self.restricciones = restricciones
        self.max_iter = max_iter
        self.vecinos = defaultdict(list)  # Diccionario para almacenar los vecinos de cada variable.
        
        # Construir lista de vecinos para cada variable según las restricciones.
        for (var1, var2) in restricciones:
            self.vecinos[var1].append(var2)
            self.vecinos[var2].append(var1)

    def num_conflictos(self, var, val, asignacion):
        """
        Calcula el número de conflictos para un valor dado.
        
        Args:
            var: Variable a evaluar.
            val: Valor asignado a la variable.
            asignacion: Asignación parcial actual.
        
        Returns:
            int: Número de conflictos generados por el valor.
        """
        return sum(1 for vecino in self.vecinos[var] 
                   if vecino in asignacion and not self.restricciones.get((var, vecino), lambda x, y: True)(val, asignacion[vecino]))

    def resolver(self):
        """
        Implementación del algoritmo de mínimos conflictos.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        # Asignación inicial aleatoria.
        asignacion = {var: random.choice(self.dominios[var]) for var in self.variables}
        
        for _ in range(self.max_iter):
            # Encontrar todas las variables con conflictos.
            vars_conflicto = [var for var in self.variables 
                              if self.num_conflictos(var, asignacion[var], asignacion) > 0]
            
            # Si no hay variables en conflicto, se encontró una solución.
            if not vars_conflicto:
                return asignacion
            
            # Seleccionar una variable aleatoria con conflictos.
            var = random.choice(vars_conflicto)
            
            # Encontrar el valor con menos conflictos.
            valores_ordenados = sorted(self.dominios[var],
                                       key=lambda val: self.num_conflictos(var, val, asignacion))
            
            # Elegir aleatoriamente entre los valores con menos conflictos.
            min_conflictos = self.num_conflictos(var, valores_ordenados[0], asignacion)
            mejores_valores = [val for val in valores_ordenados 
                               if self.num_conflictos(var, val, asignacion) == min_conflictos]
            
            # Asignar el mejor valor a la variable.
            asignacion[var] = random.choice(mejores_valores)
        
        # Si se alcanzó el máximo de iteraciones sin encontrar solución, devolver None.
        return None

def resolver_n_reinas(n=8):
    """
    Resuelve el problema de las n reinas utilizando el algoritmo de mínimos conflictos.
    
    Args:
        n: Número de reinas (y tamaño del tablero).
    
    Returns:
        bool: True si se encontró solución, False en caso contrario.
    """
    print(f"Resolviendo problema de {n} reinas con mínimos conflictos...")
    
    # Definir las variables y dominios.
    variables = [f"R{i+1}" for i in range(n)]  # Una variable por cada reina.
    dominios = {var: list(range(n)) for var in variables}  # Dominios: posiciones posibles en las filas.
    
    # Restricciones: las reinas no pueden estar en la misma fila ni en la misma diagonal.
    restricciones = {}
    for i in range(n):
        for j in range(n):
            if i != j:
                restricciones[(variables[i], variables[j])] = lambda x, y, i=i, j=j: (
                    x != y and abs(x - y) != abs(i - j))
    
    # Crear el solver de mínimos conflictos.
    solver = MinimosConflictos(variables, dominios, restricciones, max_iter=10000)
    solucion = solver.resolver()
    
    if solucion:
        # Si se encontró una solución, representarla en un tablero.
        print("\nSolución encontrada:")
        tablero = [["·" for _ in range(n)] for _ in range(n)]  # Crear un tablero vacío.
        for reina, fila in solucion.items():
            col = int(reina[1:]) - 1  # Convertir el índice de la reina a columna.
            tablero[fila][col] = "Q"  # Colocar la reina en el tablero.
        
        # Imprimir el tablero.
        for fila in tablero:
            print(" ".join(fila))
        return True
    else:
        # Si no se encontró solución, intentar nuevamente.
        print("No se encontró solución. Intentando nuevamente...")
        return False

# Ejecutar hasta encontrar solución.
while not resolver_n_reinas(8):
    pass