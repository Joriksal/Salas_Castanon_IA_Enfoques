class BacktrackingSolver:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa el solver de búsqueda de vuelta atrás (backtracking).
        
        Args:
            variables: Lista de variables a asignar.
            dominios: Diccionario {variable: lista de valores posibles}.
            restricciones: Lista de funciones que verifican restricciones (asignacion -> bool).
        """
        self.variables = variables  # Variables del problema.
        self.dominios = dominios  # Dominios posibles para cada variable.
        self.restricciones = restricciones  # Lista de restricciones a cumplir.
        self.asignacion = {}  # Asignación parcial actual (vacía al inicio).
        self.nodos_expandidos = 0  # Contador de nodos expandidos durante la búsqueda.

    def es_completa(self):
        """
        Verifica si la asignación actual incluye todas las variables.
        
        Returns:
            bool: True si todas las variables están asignadas, False en caso contrario.
        """
        # Devuelve True si todas las variables tienen un valor asignado.
        return len(self.asignacion) == len(self.variables)

    def es_consistente(self, variable, valor):
        """
        Verifica si asignar un valor a una variable es consistente con las restricciones.
        
        Args:
            variable: Variable a asignar.
            valor: Valor a asignar a la variable.
        
        Returns:
            bool: True si la asignación es consistente, False en caso contrario.
        """
        # Crear una copia temporal de la asignación para probar la consistencia.
        asignacion_temp = self.asignacion.copy()
        asignacion_temp[variable] = valor
        
        # Verificar cada restricción.
        for restriccion in self.restricciones:
            # Obtener las variables involucradas en la restricción.
            vars_restr = self._obtener_variables_restriccion(restriccion)
            # Verificar si todas las variables de la restricción están asignadas.
            if all(v in asignacion_temp for v in vars_restr):
                # Si la restricción no se cumple, devolver False.
                if not restriccion(asignacion_temp):
                    return False
        return True  # Si todas las restricciones se cumplen, devolver True.

    def _obtener_variables_restriccion(self, restriccion):
        """
        Obtiene las variables involucradas en una restricción.
        
        Args:
            restriccion: Función de restricción.
        
        Returns:
            list: Lista de variables involucradas en la restricción.
        """
        # Inspeccionar los argumentos de la función de restricción para obtener las variables.
        if hasattr(restriccion, '__code__'):
            return restriccion.__code__.co_varnames[:restriccion.__code__.co_argcount]
        return []

    def seleccionar_variable_no_asignada(self):
        """
        Selecciona la próxima variable a asignar utilizando la heurística MRV
        (Minimum Remaining Values).
        
        Returns:
            str: Variable seleccionada.
        """
        # Filtrar las variables que aún no han sido asignadas.
        no_asignadas = [v for v in self.variables if v not in self.asignacion]
        # Seleccionar la variable con el menor número de valores posibles en su dominio.
        return min(no_asignadas, key=lambda v: len(self.dominios[v]))

    def ordenar_valores(self, variable):
        """
        Ordena los valores del dominio de una variable (sin heurística adicional).
        
        Args:
            variable: Variable a asignar.
        
        Returns:
            list: Lista de valores ordenados.
        """
        # Devuelve los valores del dominio de la variable sin aplicar heurísticas.
        return self.dominios[variable]

    def resolver(self):
        """
        Algoritmo principal de búsqueda de vuelta atrás (backtracking).
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        # Reiniciar el contador de nodos expandidos antes de iniciar la búsqueda.
        self.nodos_expandidos = 0
        # Llamar a la función recursiva para resolver el problema.
        return self._backtrack()

    def _backtrack(self):
        """
        Función recursiva interna para realizar la búsqueda de vuelta atrás.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        # Incrementar el contador de nodos expandidos.
        self.nodos_expandidos += 1
        
        # Si la asignación es completa, devolver la solución.
        if self.es_completa():
            return self.asignacion.copy()

        # Seleccionar una variable no asignada.
        var = self.seleccionar_variable_no_asignada()
        
        # Probar cada valor del dominio de la variable seleccionada.
        for valor in self.ordenar_valores(var):
            # Verificar si la asignación es consistente.
            if self.es_consistente(var, valor):
                # Asignar el valor a la variable.
                self.asignacion[var] = valor
                
                # Llamada recursiva para continuar la búsqueda.
                resultado = self._backtrack()
                if resultado is not None:
                    return resultado  # Si se encuentra solución, devolverla.
                
                # Deshacer la asignación si no lleva a una solución.
                del self.asignacion[var]
        
        # Si no se encuentra solución en esta rama, devolver None.
        return None

# =============================================
# EJEMPLOS DE USO
# =============================================

if __name__ == "__main__":
    # Ejemplo 1: Problema de las 4 reinas.
    print("=== EJEMPLO 1: PROBLEMA DE LAS 4 REINAS ===")
    variables = ['Q1', 'Q2', 'Q3', 'Q4']  # Variables: una por cada reina.
    dominios = {q: [1, 2, 3, 4] for q in variables}  # Dominios: columnas posibles.

    # Restricción: ninguna reina puede atacarse.
    def no_atacan(asignacion):
        """
        Verifica que las reinas no se ataquen entre sí.
        """
        reinas = list(asignacion.items())
        for i, (q1, c1) in enumerate(reinas):
            for j, (q2, c2) in enumerate(reinas[i+1:], i+1):
                # Verificar si están en la misma columna o en diagonales.
                if c1 == c2 or abs(c1 - c2) == abs(int(q1[1]) - int(q2[1])):
                    return False
        return True

    solver = BacktrackingSolver(variables, dominios, [no_atacan])
    solucion = solver.resolver()
    
    print("Solución encontrada:", solucion)
    print("Nodos expandidos:", solver.nodos_expandidos)

    # Ejemplo 2: Sudoku 4x4.
    print("\n=== EJEMPLO 2: SUDOKU 4x4 ===")
    variables_sudoku = [f"{fila}{col}" for fila in range(4) for col in range(4)]
    
    # Grid inicial (0 = vacío).
    grid = [
        [1, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 4]
    ]
    
    dominios_sudoku = {}
    for fila in range(4):
        for col in range(4):
            var = f"{fila}{col}"
            dominios_sudoku[var] = [grid[fila][col]] if grid[fila][col] != 0 else [1, 2, 3, 4]

    # Restricción: todos los valores asignados deben ser únicos en su grupo.
    def restriccion_unica(asignacion):
        """
        Verifica que todos los valores asignados sean únicos en su grupo.
        """
        grupos = []
        
        # Filas.
        for fila in range(4):
            grupo = [asignacion.get(f"{fila}{col}") for col in range(4) 
                    if f"{fila}{col}" in asignacion]
            grupos.append(grupo)
        
        # Columnas.
        for col in range(4):
            grupo = [asignacion.get(f"{fila}{col}") for fila in range(4) 
                    if f"{fila}{col}" in asignacion]
            grupos.append(grupo)
        
        # Cuadrantes 2x2.
        for cuad_fila in range(0, 4, 2):
            for cuad_col in range(0, 4, 2):
                grupo = []
                for fila in range(cuad_fila, cuad_fila+2):
                    for col in range(cuad_col, cuad_col+2):
                        if f"{fila}{col}" in asignacion:
                            grupo.append(asignacion[f"{fila}{col}"])
                grupos.append(grupo)
        
        # Verificar unicidad en cada grupo.
        for grupo in grupos:
            valores = [v for v in grupo if v is not None]
            if len(valores) != len(set(valores)):
                return False
        return True

    solver_sudoku = BacktrackingSolver(variables_sudoku, dominios_sudoku, [restriccion_unica])
    sol_sudoku = solver_sudoku.resolver()
    
    if sol_sudoku:
        print("Solución encontrada:")
        for fila in range(4):
            print([sol_sudoku.get(f"{fila}{col}", " ") for col in range(4)])
    else:
        print("No se encontró solución")
    print("Nodos expandidos:", solver_sudoku.nodos_expandidos)