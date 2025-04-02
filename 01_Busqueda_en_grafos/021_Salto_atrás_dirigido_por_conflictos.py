class CBJ:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa el problema de satisfacción de restricciones (CSP) con salto atrás dirigido por conflictos (CBJ).
        
        Args:
            variables: Lista de variables del problema.
            dominios: Diccionario {variable: lista de valores posibles}.
            restricciones: Diccionario {(var1, var2): función} que define las restricciones entre variables.
        """
        self.variables = variables
        self.dominios = {var: list(dominios[var]) for var in variables}  # Copia de los dominios.
        self.restricciones = restricciones
        self.vecinos = {v: [] for v in variables}  # Diccionario para almacenar los vecinos de cada variable.
        
        # Construir lista de vecinos para cada variable según las restricciones.
        for (var1, var2) in restricciones:
            self.vecinos[var1].append(var2)
            self.vecinos[var2].append(var1)
    
    def consistente(self, var1, val1, var2, val2):
        """
        Verifica si dos valores son consistentes según las restricciones.
        
        Args:
            var1, var2: Variables a verificar.
            val1, val2: Valores asignados a las variables.
        
        Returns:
            bool: True si son consistentes, False en caso contrario.
        """
        if (var1, var2) in self.restricciones:
            if not self.restricciones[(var1, var2)](val1, val2):
                return False
        if (var2, var1) in self.restricciones:
            if not self.restricciones[(var2, var1)](val2, val1):
                return False
        return True
    
    def resolver(self):
        """
        Resuelve el CSP usando Conflict-Directed Backjumping (CBJ).
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        resultado = self._cbj({}, 0, {})  # Inicia la búsqueda desde el nivel 0.
        if isinstance(resultado, tuple):  # Si el resultado es un salto atrás, no hay solución.
            return None
        return resultado
    
    def _cbj(self, asignacion, nivel, conflictos_previos):
        """
        Función recursiva que implementa el algoritmo CBJ.
        
        Args:
            asignacion: Asignación parcial actual.
            nivel: Nivel actual en el árbol de búsqueda.
            conflictos_previos: Diccionario que rastrea los conflictos previos.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        # Si todas las variables están asignadas, devolver la solución.
        if len(asignacion) == len(self.variables):
            return asignacion
        
        var = self.variables[nivel]  # Seleccionar la variable correspondiente al nivel actual.
        conflict_set = set()  # Conjunto de variables en conflicto.
        
        # Probar cada valor del dominio de la variable seleccionada.
        for valor in self.dominios[var]:
            es_consistente = True
            current_conflicts = set()  # Variables en conflicto con el valor actual.
            
            # Verificar consistencia con las variables vecinas ya asignadas.
            for vecino in self.vecinos[var]:
                if vecino in asignacion:
                    if not self.consistente(var, valor, vecino, asignacion[vecino]):
                        current_conflicts.add(vecino)
                        es_consistente = False
            
            if es_consistente:
                # Si es consistente, asignar el valor y continuar con la búsqueda.
                nueva_asignacion = asignacion.copy()
                nueva_asignacion[var] = valor
                nuevos_conflictos = conflictos_previos.copy()
                
                resultado = self._cbj(nueva_asignacion, nivel + 1, nuevos_conflictos)
                if resultado is not None and not isinstance(resultado, tuple):
                    return resultado  # Solución encontrada.
            else:
                # Si no es consistente, agregar las variables en conflicto al conjunto de conflictos.
                conflict_set.update(current_conflicts)
        
        if conflict_set:
            # Si hay conflictos, encontrar el nivel más alto de las variables en conflicto.
            niveles_conflicto = []
            for v in conflict_set:
                if v in asignacion:
                    niveles_conflicto.append(self.variables.index(v))
            
            if niveles_conflicto:
                return None, max(niveles_conflicto)  # Indica salto atrás al nivel más alto de conflicto.
        
        return None  # No hay solución en este nivel.

# --------------------------------------------
# Ejemplo: Problema de las 4 Reinas con CBJ
# --------------------------------------------

if __name__ == "__main__":
    print("Resolviendo el problema de las 4 reinas con CBJ:")
    
    # Definir el problema
    variables = ['R1', 'R2', 'R3', 'R4']  # Una variable por cada reina.
    dominios = {var: [0, 1, 2, 3] for var in variables}  # Posiciones posibles en las columnas.
    
    # Restricciones: las reinas no pueden estar en la misma fila ni en la misma diagonal.
    restricciones = {}
    for i in range(4):
        for j in range(i + 1, 4):
            # Usamos una función lambda con valores por defecto para capturar i y j correctamente.
            restricciones[(variables[i], variables[j])] = (lambda x, y, i=i, j=j: 
                                                          x != y and abs(x - y) != abs(i - j))
    
    # Resolver el problema utilizando CBJ.
    problema = CBJ(variables, dominios, restricciones)
    solucion = problema.resolver()
    
    # Mostrar la solución encontrada.
    if solucion:
        print("\nSolución encontrada:")
        # Crear un tablero vacío para representar la solución.
        tablero = [["·" for _ in range(4)] for _ in range(4)]
        for reina, fila in solucion.items():
            col = int(reina[1]) - 1  # Convertir el índice de la reina a columna.
            tablero[fila][col] = "Q"  # Colocar la reina en el tablero.
        
        # Imprimir el tablero.
        for fila in tablero:
            print(" ".join(fila))
    else:
        print("No se encontró solución.")