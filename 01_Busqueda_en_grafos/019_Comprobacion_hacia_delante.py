class ForwardCheckingSolver:
    def __init__(self, variables, dominios, restricciones):
        """
        Inicializa el solver con comprobación hacia adelante (forward checking).
        
        Args:
            variables: Lista de variables.
            dominios: Diccionario {variable: lista de valores posibles}.
            restricciones: Lista de funciones que verifican restricciones (asignacion -> bool).
        """
        self.variables = variables  # Lista de variables del problema.
        self.dominios = {var: list(dominios[var]) for var in variables}  # Copia de los dominios iniciales.
        self.restricciones = restricciones  # Lista de funciones de restricciones.
        self.asignacion = {}  # Diccionario para la asignación parcial actual.
        self.nodos_expandidos = 0  # Contador de nodos expandidos durante la búsqueda.
        self.vecinos = self._construir_grafo_restricciones()  # Grafo de restricciones entre variables.

    def _construir_grafo_restricciones(self):
        """
        Crea un diccionario que mapea cada variable a sus vecinos (variables relacionadas por restricciones).
        
        Returns:
            dict: Diccionario {variable: lista de vecinos}.
        """
        vecinos = {var: set() for var in self.variables}  # Inicializa un conjunto vacío para cada variable.
        for restriccion in self.restricciones:
            vars_restr = self._obtener_variables_restriccion(restriccion)  # Obtiene las variables involucradas.
            for var in vars_restr:
                vecinos[var].update(v for v in vars_restr if v != var)  # Agrega las variables relacionadas.
        return {var: list(vecs) for var, vecs in vecinos.items()}  # Convierte los conjuntos en listas.

    def _obtener_variables_restriccion(self, restriccion):
        """
        Obtiene las variables involucradas en una restricción (función lambda).
        
        Args:
            restriccion: Función de restricción.
        
        Returns:
            list: Lista de variables involucradas en la restricción.
        """
        if hasattr(restriccion, '__code__'):  # Verifica si la restricción tiene un atributo __code__.
            return restriccion.__code__.co_varnames[:restriccion.__code__.co_argcount]  # Obtiene los nombres de las variables.
        return []

    def resolver(self):
        """
        Inicia la búsqueda con comprobación hacia adelante.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        self.nodos_expandidos = 0  # Reinicia el contador de nodos expandidos.
        return self._backtrack()  # Llama al método recursivo de backtracking.

    def _backtrack(self):
        """
        Función recursiva que realiza la búsqueda con forward checking.
        
        Returns:
            dict: Asignación completa si se encuentra solución, None en caso contrario.
        """
        self.nodos_expandidos += 1  # Incrementa el contador de nodos expandidos.

        # Si todas las variables están asignadas, devuelve la solución.
        if len(self.asignacion) == len(self.variables):
            return self.asignacion.copy()

        # Selecciona la siguiente variable a asignar utilizando la heurística MRV.
        var = self._seleccionar_variable()

        # Prueba cada valor del dominio de la variable seleccionada.
        for valor in self._ordenar_valores(var):
            if self._es_consistente(var, valor):  # Verifica consistencia antes de asignar.
                # Hace una copia de los dominios antes de modificarlos.
                dominios_originales = {v: list(self.dominios[v]) for v in self.variables}

                # Asigna el valor a la variable.
                self.asignacion[var] = valor

                # Aplica forward checking para reducir dominios de vecinos.
                reducciones = self._forward_checking(var, valor)

                if reducciones is not None:  # Si no hay dominios vacíos, continúa.
                    resultado = self._backtrack()
                    if resultado is not None:
                        return resultado  # Solución encontrada.

                # Restaura dominios y deshace la asignación si la rama falla.
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
        asignacion_temp = self.asignacion.copy()  # Crea una copia de la asignación actual.
        asignacion_temp[variable] = valor  # Asigna temporalmente el valor.
        return self._cumple_restricciones(asignacion_temp)  # Verifica si cumple las restricciones.

    def _forward_checking(self, variable, valor):
        """
        Reduce los dominios de las variables vecinas después de asignar un valor.
        
        Args:
            variable: Variable asignada.
            valor: Valor asignado.
        
        Returns:
            dict: Reducciones realizadas en los dominios, o None si algún dominio queda vacío.
        """
        reducciones = {}  # Guarda los valores eliminados de los dominios.

        for vecina in self.vecinos[variable]:
            if vecina not in self.asignacion:  # Solo considera variables no asignadas.
                for val in self.dominios[vecina][:]:  # Itera sobre una copia del dominio.
                    asignacion_temp = self.asignacion.copy()
                    asignacion_temp[vecina] = val
                    asignacion_temp[variable] = valor

                    if not self._cumple_restricciones(asignacion_temp):  # Verifica restricciones.
                        if vecina not in reducciones:
                            reducciones[vecina] = []
                        reducciones[vecina].append(val)
                        self.dominios[vecina].remove(val)  # Elimina el valor inconsistente.

                if not self.dominios[vecina]:  # Si el dominio queda vacío, devuelve None.
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
            vars_restr = self._obtener_variables_restriccion(restriccion)  # Variables involucradas.
            if all(v in asignacion for v in vars_restr):  # Verifica si todas las variables están asignadas.
                args = {v: asignacion[v] for v in vars_restr}  # Crea un diccionario con los valores asignados.
                if not restriccion(**args):  # Evalúa la restricción.
                    return False
        return True

    def _seleccionar_variable(self):
        """
        Selecciona la próxima variable a asignar utilizando la heurística MRV (Minimum Remaining Values).
        
        Returns:
            str: Variable seleccionada.
        """
        no_asignadas = [v for v in self.variables if v not in self.asignacion]  # Variables no asignadas.
        return min(no_asignadas, key=lambda v: len(self.dominios[v]))  # Selecciona la de menor dominio.

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
            if vecina not in self.asignacion:  # Solo considera variables no asignadas.
                for val in self.dominios[vecina]:
                    asignacion_temp = {variable: valor, vecina: val}
                    if not self._cumple_restricciones(asignacion_temp):  # Verifica restricciones.
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