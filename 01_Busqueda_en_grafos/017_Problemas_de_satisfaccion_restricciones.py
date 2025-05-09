class CSP:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa un problema CSP (Problema de Satisfacción de Restricciones).
        
        Args:
            variables: Lista de nombres de variables.
            dominios: Diccionario {variable: lista_de_valores_posibles}.
            restricciones: Lista de funciones que verifican restricciones (asignacion -> bool).
        """
        # Lista de variables del problema
        self.variables = variables
        # Copia de los dominios para cada variable
        self.dominios = {var: list(dominios[var]) for var in variables}
        # Lista de restricciones (funciones lambda que devuelven True/False)
        self.restricciones = restricciones
        # Diccionario para almacenar los vecinos de cada variable
        self.vecinos = {var: [] for var in variables}
        
        # Preprocesar vecinos para cada variable según las restricciones
        for restriccion in restricciones:
            vars_involucradas = self._obtener_variables_en_restriccion(restriccion)
            for var in vars_involucradas:
                # Agregar como vecinos las otras variables involucradas en la restricción
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
        # Verifica si la restricción tiene un atributo '__code__' (es una función)
        if hasattr(restriccion, '__code__'):
            # Obtiene los nombres de las variables utilizadas en la función
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
        # Crear una copia temporal de la asignación actual
        asignacion_temporal = asignacion.copy()
        asignacion_temporal[variable] = valor
        
        # Verificar cada restricción
        for restriccion in self.restricciones:
            # Obtener las variables involucradas en la restricción
            vars_restr = self._obtener_variables_en_restriccion(restriccion)
            if variable in vars_restr:
                # Verificar si todas las variables de la restricción están asignadas
                if all(v in asignacion_temporal for v in vars_restr):
                    # Evaluar la restricción con los valores asignados
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
        # Filtrar variables que aún no han sido asignadas
        no_asignadas = [v for v in self.variables if v not in asignacion]
        # Seleccionar la variable con el menor número de valores posibles en su dominio
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
        # Retorna los valores del dominio ordenados (sin heurística adicional)
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
        # Diccionario para registrar reducciones en los dominios
        reducciones = {}
        
        # Iterar sobre las variables vecinas
        for vecina in self.vecinos[variable]:
            if vecina not in asignacion:
                for val in self.dominios[vecina][:]:
                    # Crear asignación temporal para probar consistencia
                    asignacion_temporal = asignacion.copy()
                    asignacion_temporal[variable] = valor
                    asignacion_temporal[vecina] = val
                    
                    # Si no es consistente, eliminar el valor del dominio
                    if not self.es_consistente(vecina, val, asignacion_temporal):
                        if vecina not in reducciones:
                            reducciones[vecina] = []
                        reducciones[vecina].append(val)
                        self.dominios[vecina].remove(val)
                
                # Si el dominio de una vecina queda vacío, revertir cambios
                if not self.dominios[vecina]:
                    self._revertir_reducciones(reducciones)
                    return False
        
        return reducciones
    
    def _revertir_reducciones(self, reducciones):
        """
        Revierte los cambios hechos por forward checking.
        
        Args:
            reducciones: Cambios realizados en los dominios.
        """
        # Restaurar los valores eliminados en los dominios
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
        # Si todas las variables están asignadas, se encontró una solución
        if len(asignacion) == len(self.variables):
            return asignacion
        
        # Seleccionar una variable no asignada
        var = self.seleccionar_variable_no_asignada(asignacion)
        
        # Probar cada valor en el dominio de la variable
        for valor in self.ordenar_valores(var, asignacion):
            if self.es_consistente(var, valor, asignacion):
                # Crear una nueva asignación parcial
                nueva_asignacion = asignacion.copy()
                nueva_asignacion[var] = valor
                
                # Hacer copia de dominios antes de aplicar forward checking
                dominios_originales = {v: list(self.dominios[v]) for v in self.variables}
                reducciones = self.forward_checking(var, valor, nueva_asignacion)
                
                if reducciones is not False:  # Si no hubo dominios vacíos
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
    # Ejemplo de resolución de un Sudoku simplificado (4x4)
    variables_sudoku = [f"{fila}{col}" for fila in range(4) for col in range(4)]
    
    # Grid inicial (0 = vacío)
    grid = [
        [1, 0, 0, 0],
        [0, 0, 3, 0],
        [0, 2, 0, 0],
        [0, 0, 0, 4]
    ]
    
    # Crear dominios para cada celda del Sudoku
    dominios_sudoku = {}
    for fila in range(4):
        for col in range(4):
            var = f"{fila}{col}"
            dominios_sudoku[var] = [grid[fila][col]] if grid[fila][col] != 0 else [1, 2, 3, 4]

    # Generar restricciones para el Sudoku
    def generar_restricciones_sudoku():
        restricciones = []
        
        # Restricciones por fila
        for fila in range(4):
            restricciones.append(lambda **vals: len(set(vals.values())) == len(vals))
        
        # Restricciones por columna
        for col in range(4):
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

    # Crear y resolver el CSP para el Sudoku
    csp_sudoku = CSP(variables_sudoku, dominios_sudoku, generar_restricciones_sudoku())
    sol_sudoku = csp_sudoku.resolver()
    
    # Mostrar solución en formato de grid
    if sol_sudoku:
        print("Solución Sudoku 4x4:")
        for fila in range(4):
            print([sol_sudoku.get(f"{fila}{col}", 0) for col in range(4)])
    else:
        print("No se encontró solución")