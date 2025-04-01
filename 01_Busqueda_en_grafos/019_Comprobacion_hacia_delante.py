class ForwardCheckingSolver:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa el solver con comprobación hacia adelante (forward checking).
        
        Args:
            variables: Lista de variables.
            dominios: Diccionario {variable: lista de valores posibles}.
            restricciones: Lista de funciones que verifican restricciones (asignacion -> bool).
        """
        self.variables = variables
        self.dominios = {var: list(dominios[var]) for var in variables}  # Copia de los dominios.
        self.restricciones = restricciones
        self.asignacion = {}  # Asignación parcial actual.
        self.nodos_expandidos = 0  # Contador de nodos expandidos.
        self.vecinos = self._construir_grafo_restricciones()  # Construir grafo de restricciones.

    def _construir_grafo_restricciones(self):
        """
        Crea un diccionario que mapea cada variable a sus vecinos (variables relacionadas por restricciones).
        
        Returns:
            dict: Diccionario {variable: lista de vecinos}.
        """
        vecinos = {var: set() for var in self.variables}
        for restriccion in self.restricciones:
            vars_restr = self._obtener_variables_restriccion(restriccion)
            for var in vars_restr:
                vecinos[var].update(v for v in vars_restr if v != var)
        return {var: list(vecs) for var, vecs in vecinos.items()}

    def _obtener_variables_restriccion(self, restriccion):
        """
        Obtiene las variables involucradas en una restricción (función lambda).
        
        Args:
            restriccion: Función de restricción.
        
        Returns:
            list: Lista de variables involucradas en la restricción.
        """
        if hasattr(restriccion, '__code__'):
            return restriccion.__code__.co_varnames[:restriccion.__code__.co_argcount]
        return []

    def resolver(self):
        """
        Inicia la búsqueda con comprobación hacia adelante.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        self.nodos_expandidos = 0  # Reiniciar contador de nodos expandidos.
        return self._backtrack()

    def _backtrack(self):
        """
        Función recursiva que realiza la búsqueda con forward checking.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        self.nodos_expandidos += 1  # Incrementar contador de nodos expandidos.

        # Si todas las variables están asignadas, devolver la solución.
        if len(self.asignacion) == len(self.variables):
            return self.asignacion.copy()

        # Seleccionar la siguiente variable a asignar utilizando MRV.
        var = self._seleccionar_variable()

        # Probar cada valor del dominio de la variable seleccionada.
        for valor in self._ordenar_valores(var):
            if self._es_consistente(var, valor):  # Verificar consistencia antes de asignar.
                # Hacer copia de dominios antes de modificarlos.
                dominios_originales = {v: list(self.dominios[v]) for v in self.variables}

                # Asignar el valor a la variable.
                self.asignacion[var] = valor

                # Aplicar forward checking para reducir dominios de vecinos.
                reducciones = self._forward_checking(var, valor)

                if reducciones is not None:  # Si no hay dominios vacíos, continuar.
                    resultado = self._backtrack()
                    if resultado is not None:
                        return resultado  # Solución encontrada.

                # Restaurar dominios y deshacer asignación si la rama falla.
                self.dominios = dominios_originales
                del self.asignacion[var]

        return None  # No se encontró solución en esta rama.

    def _es_consistente(self, variable, valor):
        """
        Verifica si asignar un valor a una variable es consistente con las restricciones.
        
        Args:
            variable: Variable a asignar.
            valor: Valor a asignar.
        
        Returns:
            bool: True si es consistente, False en caso contrario.
        """
        asignacion_temp = self.asignacion.copy()
        asignacion_temp[variable] = valor
        return self._cumple_restricciones(asignacion_temp)

    def _forward_checking(self, variable, valor):
        """
        Reduce los dominios de las variables vecinas después de asignar un valor.
        
        Args:
            variable: Variable asignada.
            valor: Valor asignado.
        
        Returns:
            dict: Reducciones realizadas en los dominios, o None si algún dominio queda vacío.
        """
        reducciones = {}  # Guardar valores eliminados de los dominios.

        for vecina in self.vecinos[variable]:
            if vecina not in self.asignacion:
                for val in self.dominios[vecina][:]:  # Iterar sobre una copia del dominio.
                    asignacion_temp = self.asignacion.copy()
                    asignacion_temp[vecina] = val
                    asignacion_temp[variable] = valor

                    if not self._cumple_restricciones(asignacion_temp):
                        if vecina not in reducciones:
                            reducciones[vecina] = []
                        reducciones[vecina].append(val)
                        self.dominios[vecina].remove(val)

                if not self.dominios[vecina]:  # Si el dominio queda vacío, devolver None.
                    return None

        return reducciones

    def _cumple_restricciones(self, asignacion):
        """
        Verifica si una asignación cumple todas las restricciones aplicables.
        
        Args:
            asignacion: Asignación parcial.
        
        Returns:
            bool: True si cumple todas las restricciones, False en caso contrario.
        """
        for restriccion in self.restricciones:
            vars_restr = self._obtener_variables_restriccion(restriccion)
            if all(v in asignacion for v in vars_restr):
                args = {v: asignacion[v] for v in vars_restr}
                if not restriccion(**args):
                    return False
        return True

    def _seleccionar_variable(self):
        """
        Selecciona la próxima variable a asignar utilizando la heurística MRV (Minimum Remaining Values).
        
        Returns:
            str: Variable seleccionada.
        """
        no_asignadas = [v for v in self.variables if v not in self.asignacion]
        return min(no_asignadas, key=lambda v: len(self.dominios[v]))

    def _ordenar_valores(self, variable):
        """
        Ordena los valores del dominio de una variable utilizando la heurística LCV (Least Constraining Value).
        
        Args:
            variable: Variable a asignar.
        
        Returns:
            list: Lista de valores ordenados.
        """
        return sorted(self.dominios[variable], key=lambda v: self._contar_conflictos(variable, v))

    def _contar_conflictos(self, variable, valor):
        """
        Cuenta cuántas opciones eliminaría este valor en los dominios de las variables vecinas.
        
        Args:
            variable: Variable a asignar.
            valor: Valor a probar.
        
        Returns:
            int: Número de conflictos generados.
        """
        count = 0
        for vecina in self.vecinos[variable]:
            if vecina not in self.asignacion:
                for val in self.dominios[vecina]:
                    asignacion_temp = {variable: valor, vecina: val}
                    if not self._cumple_restricciones(asignacion_temp):
                        count += 1
        return count

# =============================================
# EJEMPLOS DE USO
# =============================================

if __name__ == "__main__":
    print("=== EJEMPLO 1: PROBLEMA DE LAS 4 REINAS ===")
    variables = ['Q1', 'Q2', 'Q3', 'Q4']
    dominios = {q: [1, 2, 3, 4] for q in variables}

    restricciones = [
        lambda Q1, Q2: Q1 != Q2 and abs(Q1 - Q2) != 1,
        lambda Q1, Q3: Q1 != Q3 and abs(Q1 - Q3) != 2,
        lambda Q1, Q4: Q1 != Q4 and abs(Q1 - Q4) != 3,
        lambda Q2, Q3: Q2 != Q3 and abs(Q2 - Q3) != 1,
        lambda Q2, Q4: Q2 != Q4 and abs(Q2 - Q4) != 2,
        lambda Q3, Q4: Q3 != Q4 and abs(Q3 - Q4) != 1
    ]

    solver = ForwardCheckingSolver(variables, dominios, restricciones)
    solucion = solver.resolver()
    print("Solución encontrada:", solucion)
    print("Nodos expandidos:", solver.nodos_expandidos)

    print("\n=== EJEMPLO 2: PROBLEMA DE COLORACIÓN DE MAPAS ===")
    variables = ['WA', 'NT', 'SA', 'Q', 'NSW', 'V', 'T']
    dominios = {var: ['R', 'G', 'B'] for var in variables}
    restricciones = [
        lambda WA, NT: WA != NT,
        lambda WA, SA: WA != SA,
        lambda NT, SA: NT != SA,
        lambda NT, Q: NT != Q,
        lambda SA, Q: SA != Q,
        lambda SA, NSW: SA != NSW,
        lambda SA, V: SA != V,
        lambda Q, NSW: Q != NSW,
        lambda NSW, V: NSW != V
    ]

    solver_mapas = ForwardCheckingSolver(variables, dominios, restricciones)
    solucion_mapa = solver_mapas.resolver()
    print("Coloración encontrada:")
    for region, color in solucion_mapa.items():
        print(f"{region}: {color}")
    print("Nodos expandidos:", solver_mapas.nodos_expandidos)