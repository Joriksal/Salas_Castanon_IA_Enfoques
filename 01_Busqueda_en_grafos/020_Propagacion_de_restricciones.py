from collections import deque  # Importa deque para manejar una cola eficiente en el algoritmo AC-3.

class CSP:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa un problema de satisfacción de restricciones (CSP).
        
        Args:
            variables: Lista de variables.
            dominios: Diccionario {variable: lista de valores posibles}.
            restricciones: Diccionario {(var1, var2): función} que define las restricciones entre variables.
        """
        self.variables = variables  # Lista de variables del problema.
        self.dominios = {var: list(dominios[var]) for var in variables}  # Copia de los dominios para evitar modificar el original.
        self.restricciones = restricciones  # Diccionario de restricciones entre pares de variables.
        self.vecinos = {v: [] for v in variables}  # Diccionario que almacena los vecinos de cada variable.
        
        # Construir el grafo de restricciones (vecinos).
        for (var1, var2) in restricciones:
            self.vecinos[var1].append(var2)  # Agregar var2 como vecino de var1.
            self.vecinos[var2].append(var1)  # Agregar var1 como vecino de var2.

    def consistente(self, var1, val1, var2, val2):
        """
        Verifica si dos valores son consistentes con las restricciones.
        
        Args:
            var1, var2: Variables a verificar.
            val1, val2: Valores asignados a las variables.
        
        Returns:
            bool: True si son consistentes, False en caso contrario.
        """
        # Verifica si existe una restricción entre var1 y var2 y si los valores no la cumplen.
        if (var1, var2) in self.restricciones and not self.restricciones[(var1, var2)](val1, val2):
            return False
        # Verifica la restricción en el sentido inverso (var2, var1).
        if (var2, var1) in self.restricciones and not self.restricciones[(var2, var1)](val2, val1):
            return False
        return True  # Si pasa ambas verificaciones, los valores son consistentes.

    def forward_checking(self, asignacion, var, valor):
        """
        Aplica forward checking para reducir los dominios de las variables vecinas.
        
        Args:
            asignacion: Asignación parcial actual.
            var: Variable asignada.
            valor: Valor asignado a la variable.
        
        Returns:
            dict: Reducciones realizadas en los dominios.
        """
        reducciones = {}  # Diccionario para registrar los valores eliminados de los dominios.
        for vecino in self.vecinos[var]:  # Iterar sobre los vecinos de la variable asignada.
            if vecino not in asignacion:  # Solo considerar vecinos no asignados.
                for val in list(self.dominios[vecino]):  # Iterar sobre una copia del dominio del vecino.
                    if not self.consistente(var, valor, vecino, val):  # Verificar consistencia.
                        if vecino not in reducciones:
                            reducciones[vecino] = []  # Inicializar lista de reducciones para el vecino.
                        reducciones[vecino].append(val)  # Registrar el valor eliminado.
                        self.dominios[vecino].remove(val)  # Eliminar el valor del dominio.
        return reducciones

    def revertir_reducciones(self, reducciones):
        """
        Revierte los cambios realizados por forward checking.
        
        Args:
            reducciones: Cambios realizados en los dominios.
        """
        for var in reducciones:  # Iterar sobre las variables afectadas.
            for val in reducciones[var]:  # Restaurar cada valor eliminado.
                self.dominios[var].append(val)

    def backtracking_search(self, asignacion={}):
        """
        Realiza la búsqueda de vuelta atrás con forward checking.
        
        Args:
            asignacion: Asignación parcial inicial.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        if len(asignacion) == len(self.variables):  # Si todas las variables están asignadas, devolver solución.
            return asignacion
        
        # Seleccionar la siguiente variable a asignar.
        var = self.seleccionar_variable_no_asignada(asignacion)
        
        # Probar cada valor del dominio de la variable seleccionada.
        for valor in self.ordenar_valores(var, asignacion):
            if self.consistente_con_asignacion(var, valor, asignacion):  # Verificar consistencia con la asignación actual.
                asignacion[var] = valor  # Asignar valor.
                reducciones = self.forward_checking(asignacion, var, valor)  # Aplicar forward checking.
                resultado = self.backtracking_search(asignacion)  # Llamada recursiva.
                if resultado is not None:  # Si se encuentra solución, devolverla.
                    return resultado
                del asignacion[var]  # Deshacer asignación si la rama falla.
                self.revertir_reducciones(reducciones)  # Revertir cambios en los dominios.
        return None  # No se encontró solución.

    def seleccionar_variable_no_asignada(self, asignacion):
        """
        Selecciona la próxima variable no asignada utilizando la heurística MRV (Minimum Remaining Values).
        
        Args:
            asignacion: Asignación parcial actual.
        
        Returns:
            str: Variable seleccionada.
        """
        no_asignadas = [v for v in self.variables if v not in asignacion]  # Filtrar variables no asignadas.
        return min(no_asignadas, key=lambda v: len(self.dominios[v]))  # Seleccionar la variable con menor dominio.

    def ordenar_valores(self, var, asignacion):
        """
        Ordena los valores del dominio de una variable utilizando la heurística LCV (Least Constraining Value).
        
        Args:
            var: Variable a asignar.
            asignacion: Asignación parcial actual.
        
        Returns:
            list: Lista de valores ordenados.
        """
        return sorted(self.dominios[var], key=lambda val: self.num_conflictos(var, val, asignacion))

    def num_conflictos(self, var, val, asignacion):
        """
        Cuenta el número de conflictos que genera un valor en los vecinos de una variable.
        
        Args:
            var: Variable a asignar.
            val: Valor a probar.
            asignacion: Asignación parcial actual.
        
        Returns:
            int: Número de conflictos generados.
        """
        return sum(1 for vecino in self.vecinos[var] 
                   if vecino in asignacion and not self.consistente(var, val, vecino, asignacion[vecino]))

    def consistente_con_asignacion(self, var, val, asignacion):
        """
        Verifica si un valor es consistente con la asignación actual.
        
        Args:
            var: Variable a asignar.
            val: Valor a probar.
            asignacion: Asignación parcial actual.
        
        Returns:
            bool: True si es consistente, False en caso contrario.
        """
        return all(self.consistente(var, val, v, asignacion[v]) for v in asignacion if v in self.vecinos[var])

    def AC3(self):
        """
        Aplica el algoritmo AC-3 para garantizar la consistencia de arcos.
        
        Returns:
            bool: True si el problema es consistente, False si algún dominio queda vacío.
        """
        cola = deque()  # Cola para manejar los arcos pendientes de revisión.
        for (xi, xj) in self.restricciones:  # Inicializar la cola con todos los arcos.
            cola.append((xi, xj))
            cola.append((xj, xi))
        
        while cola:  # Mientras haya arcos en la cola.
            xi, xj = cola.popleft()  # Extraer un arco de la cola.
            if self.revisar_arco(xi, xj):  # Revisar el arco.
                if not self.dominios[xi]:  # Si algún dominio queda vacío, el problema no es consistente.
                    return False
                for xk in self.vecinos[xi]:  # Agregar los vecinos de xi a la cola.
                    if xk != xj:
                        cola.append((xk, xi))
        return True  # El problema es consistente.

    def revisar_arco(self, xi, xj):
        """
        Revisa y ajusta el arco entre dos variables para garantizar consistencia.
        
        Args:
            xi, xj: Variables a revisar.
        
        Returns:
            bool: True si se realizaron cambios, False en caso contrario.
        """
        revisado = False  # Indica si se realizaron cambios en el dominio de xi.
        for x in list(self.dominios[xi]):  # Iterar sobre una copia del dominio de xi.
            if not any(self.consistente(xi, x, xj, y) for y in self.dominios[xj]):  # Verificar consistencia.
                self.dominios[xi].remove(x)  # Eliminar el valor inconsistente.
                revisado = True
        return revisado

def resolver_sudoku(tablero):
    """
    Resuelve un Sudoku utilizando propagación de restricciones y búsqueda de vuelta atrás.
    
    Args:
        tablero: Lista de listas que representa el tablero del Sudoku (0 = vacío).
    """
    variables = [f"C{i+1}{j+1}" for i in range(9) for j in range(9)]  # Generar variables para cada celda.
    dominios = {}
    restricciones = {}
    
    # Inicializar dominios.
    for i in range(9):
        for j in range(9):
            var = f"C{i+1}{j+1}"
            dominios[var] = [tablero[i][j]] if tablero[i][j] != 0 else list(range(1, 10))
    
    # Restricciones para filas y columnas.
    for i in range(9):
        for j1 in range(9):
            for j2 in range(j1 + 1, 9):
                var1, var2 = f"C{i+1}{j1+1}", f"C{i+1}{j2+1}"
                restricciones[(var1, var2)] = lambda x, y: x != y
                var1, var2 = f"C{j1+1}{i+1}", f"C{j2+1}{i+1}"
                restricciones[(var1, var2)] = lambda x, y: x != y
    
    # Restricciones para cajas 3x3.
    for box_i in range(3):
        for box_j in range(3):
            for pos1 in range(9):
                for pos2 in range(pos1 + 1, 9):
                    i1, j1 = box_i * 3 + pos1 // 3, box_j * 3 + pos1 % 3
                    i2, j2 = box_i * 3 + pos2 // 3, box_j * 3 + pos2 % 3
                    var1, var2 = f"C{i1+1}{j1+1}", f"C{i2+1}{j2+1}"
                    restricciones[(var1, var2)] = lambda x, y: x != y
    
    problema = CSP(variables, dominios, restricciones)  # Crear el problema CSP.
    problema.AC3()  # Aplicar AC-3 para reducir dominios.
    solucion = problema.backtracking_search()  # Resolver con búsqueda de vuelta atrás.
    
    if solucion:
        print("\nSudoku resuelto:")
        sudoku_resuelto = [[0 for _ in range(9)] for _ in range(9)]
        for var, val in solucion.items():
            i, j = int(var[1]) - 1, int(var[2]) - 1
            sudoku_resuelto[i][j] = val
        for fila in sudoku_resuelto:
            print(" ".join(map(str, fila)))
    else:
        print("No se encontró solución.")

# Ejemplo de Sudoku difícil.
sudoku = [
    [5, 3, 0, 0, 7, 0, 0, 0, 0],
    [6, 0, 0, 1, 9, 5, 0, 0, 0],
    [0, 9, 8, 0, 0, 0, 0, 6, 0],
    [8, 0, 0, 0, 6, 0, 0, 0, 3],
    [4, 0, 0, 8, 0, 3, 0, 0, 1],
    [7, 0, 0, 0, 2, 0, 0, 0, 6],
    [0, 6, 0, 0, 0, 0, 2, 8, 0],
    [0, 0, 0, 4, 1, 9, 0, 0, 5],
    [0, 0, 0, 0, 8, 0, 0, 7, 9]
]

resolver_sudoku(sudoku)