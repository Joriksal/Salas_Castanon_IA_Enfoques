class CSP:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa un problema CSP (Problema de Satisfacción de Restricciones).
        
        Args:
            variables: Lista de nombres de variables.
            dominios: Diccionario {variable: lista_de_valores_posibles}.
            restricciones: Lista de funciones que verifican restricciones (asignacion -> bool).
        """
        self.variables = variables
        self.dominios = {var: list(dominios[var]) for var in variables}  # Copia de los dominios.
        self.restricciones = restricciones
        self.vecinos = {var: [] for var in variables}
        
        # Preprocesar vecinos para cada variable
        for restriccion in restricciones:
            vars_involucradas = self._obtener_variables_en_restriccion(restriccion)
            for var in vars_involucradas:
                self.vecinos[var].extend([v for v in vars_involucradas if v != var])
        
        # Eliminar duplicados en la lista de vecinos
        self.vecinos = {var: list(set(vecs)) for var, vecs in self.vecinos.items()}
    
    def _obtener_variables_en_restriccion(self, restriccion):
        """
        Obtiene las variables involucradas en una restricción.
        Asume que las restricciones son funciones lambda que usan como parámetros
        solo las variables relevantes.
        
        Args:
            restriccion: Función de restricción.
        
        Returns:
            list: Lista de nombres de variables utilizadas en la restricción.
        """
        if hasattr(restriccion, '__code__'):
            return list(restriccion.__code__.co_varnames[:restriccion.__code__.co_argcount])
        return []
    
    def es_consistente(self, variable, valor, asignacion):
        """
        Verifica si una asignación parcial es consistente con las restricciones.
        
        Args:
            variable: Variable a asignar.
            valor: Valor a asignar a la variable.
            asignacion: Asignación parcial actual.
        
        Returns:
            bool: True si la asignación es consistente, False en caso contrario.
        """
        asignacion_temporal = asignacion.copy()
        asignacion_temporal[variable] = valor
        
        for restriccion in self.restricciones:
            # Verificar solo restricciones que involucren la variable actual
            vars_restr = self._obtener_variables_en_restriccion(restriccion)
            if variable in vars_restr:
                # Verificar si todas las variables de la restricción están asignadas
                if all(v in asignacion_temporal for v in vars_restr):
                    if not restriccion(**{v: asignacion_temporal[v] for v in vars_restr}):
                        return False
        return True
    
    def seleccionar_variable_no_asignada(self, asignacion):
        """
        Selecciona una variable no asignada utilizando la heurística MRV
        (Minimum Remaining Values).
        
        Args:
            asignacion: Asignación parcial actual.
        
        Returns:
            str: Variable seleccionada.
        """
        no_asignadas = [v for v in self.variables if v not in asignacion]
        return min(no_asignadas, key=lambda var: len(self.dominios[var]))
    
    def ordenar_valores(self, variable, asignacion):
        """
        Ordena los valores del dominio de una variable (heurística: menor restricción primero).
        
        Args:
            variable: Variable a asignar.
            asignacion: Asignación parcial actual.
        
        Returns:
            list: Lista de valores ordenados.
        """
        return sorted(self.dominios[variable])
    
    def forward_checking(self, variable, valor, asignacion):
        """
        Aplica forward checking para reducir los dominios de las variables vecinas.
        
        Args:
            variable: Variable asignada.
            valor: Valor asignado a la variable.
            asignacion: Asignación parcial actual.
        
        Returns:
            dict: Reducciones realizadas en los dominios (para revertir si es necesario).
        """
        reducciones = {}  # Guardar cambios para poder revertirlos
        
        for vecina in self.vecinos[variable]:
            if vecina not in asignacion:
                for val in self.dominios[vecina][:]:
                    # Crear asignación temporal para probar consistencia
                    asignacion_temporal = asignacion.copy()
                    asignacion_temporal[variable] = valor
                    asignacion_temporal[vecina] = val
                    
                    if not self.es_consistente(vecina, val, asignacion_temporal):
                        if vecina not in reducciones:
                            reducciones[vecina] = []
                        reducciones[vecina].append(val)
                        self.dominios[vecina].remove(val)
                
                if not self.dominios[vecina]:  # Dominio vacío
                    self._revertir_reducciones(reducciones)
                    return False
        
        return reducciones
    
    def _revertir_reducciones(self, reducciones):
        """
        Revierte los cambios hechos por forward checking.
        
        Args:
            reducciones: Cambios realizados en los dominios.
        """
        for var, valores in reducciones.items():
            self.dominios[var].extend(valores)
    
    def resolver(self, asignacion={}):
        """
        Resuelve el CSP utilizando backtracking con MRV y forward checking.
        
        Args:
            asignacion: Asignación parcial inicial (vacía por defecto).
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        if len(asignacion) == len(self.variables):
            return asignacion  # Solución encontrada.
        
        var = self.seleccionar_variable_no_asignada(asignacion)
        
        for valor in self.ordenar_valores(var, asignacion):
            if self.es_consistente(var, valor, asignacion):
                nueva_asignacion = asignacion.copy()
                nueva_asignacion[var] = valor
                
                # Hacer copia de dominios antes de forward checking
                dominios_originales = {v: list(self.dominios[v]) for v in self.variables}
                reducciones = self.forward_checking(var, valor, nueva_asignacion)
                
                if reducciones is not False:  # No hubo dominios vacíos
                    resultado = self.resolver(nueva_asignacion)
                    if resultado is not None:
                        return resultado
                
                # Revertir cambios si la rama falló
                self.dominios = dominios_originales
        
        return None

# =============================================
# EJEMPLOS DE USO
# =============================================

if __name__ == "__main__":
    print("\n=== EJEMPLO 2: SUDOKU SIMPLIFICADO (4x4) ===")
    variables_sudoku = [f"{fila}{col}" for fila in range(4) for col in range(4)]
    
    # Grid inicial (0 = vacío)
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

    # Restricciones para Sudoku
    def generar_restricciones_sudoku():
        restricciones = []
        
        # Restricciones por fila
        for fila in range(4):
            vars_fila = [f"{fila}{col}" for col in range(4)]
            restricciones.append(lambda **vals: len(set(vals.values())) == len(vals))
        
        # Restricciones por columna
        for col in range(4):
            vars_col = [f"{fila}{col}" for fila in range(4)]
            restricciones.append(lambda **vals: len(set(vals.values())) == len(vals))
        
        # Restricciones por cuadrante 2x2
        cuadrantes = [
            ['00', '01', '10', '11'],
            ['02', '03', '12', '13'],
            ['20', '21', '30', '31'],
            ['22', '23', '32', '33']
        ]
        for cuadrante in cuadrantes:
            restricciones.append(lambda **vals: len(set(vals.values())) == len(vals))
        
        return restricciones

    csp_sudoku = CSP(variables_sudoku, dominios_sudoku, generar_restricciones_sudoku())
    sol_sudoku = csp_sudoku.resolver()
    
    # Mostrar solución en formato de grid
    if sol_sudoku:
        print("Solución Sudoku 4x4:")
        for fila in range(4):
            print([sol_sudoku.get(f"{fila}{col}", 0) for col in range(4)])
    else:
        print("No se encontró solución")