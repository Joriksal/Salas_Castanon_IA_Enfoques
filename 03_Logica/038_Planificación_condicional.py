class ConditionalPlanner:
    def __init__(self):
        # Define los operadores disponibles con sus precondiciones y efectos
        self.operators = {
            "MoverCaja": {
                "preconditions": ["CajaEnA", "RobotEnA"],  # Condiciones necesarias para ejecutar la acción
                "effects": [
                    {"condition": [], "add": ["CajaEnB"], "remove": ["CajaEnA"]},  # Efecto deseado
                    {"condition": ["SueloMojado"], "add": ["CajaResbalo"], "remove": []}  # Efecto no deseado
                ]
            },
            "SecarSuelo": {
                "preconditions": ["SueloMojado", "RobotEnA"],  # Condiciones necesarias para secar el suelo
                "effects": [
                    {"condition": [], "add": [], "remove": ["SueloMojado"]}  # Elimina la condición de suelo mojado
                ]
            }
        }

    def apply_effects(self, state, effects):
        """
        Aplica los efectos condicionales de una acción al estado actual.
        Devuelve el nuevo estado y un indicador de si hubo efectos no deseados.
        """
        new_state = set(state)  # Copia del estado actual
        undesired_effects = False  # Bandera para detectar efectos no deseados

        for effect in effects:
            # Verifica si se cumplen las condiciones para aplicar el efecto
            if all(cond in new_state for cond in effect["condition"]):
                new_state.update(effect["add"])  # Agrega los efectos positivos
                new_state.difference_update(effect["remove"])  # Elimina los efectos negativos
                if "CajaResbalo" in effect["add"]:  # Detecta si se generó un efecto no deseado
                    undesired_effects = True

        return new_state, undesired_effects

    def plan(self, goal, initial_state):
        """
        Genera un plan para alcanzar el objetivo desde el estado inicial,
        evitando efectos no deseados.
        """
        from collections import deque

        queue = deque()  # Cola para realizar búsqueda en anchura
        queue.append((list(initial_state), []))  # (estado actual, plan parcial)
        visited = set()  # Conjunto de estados visitados para evitar ciclos

        while queue:
            current_state, current_plan = queue.popleft()  # Extrae el siguiente nodo
            state_tuple = frozenset(current_state)  # Convierte el estado a un formato inmutable

            if state_tuple in visited:
                continue  # Si ya se visitó este estado, lo ignora
            visited.add(state_tuple)

            # Verifica si el objetivo está cumplido
            if all(g in current_state for g in goal):
                return current_plan  # Devuelve el plan si se cumple el objetivo

            # Genera sucesores aplicando operadores
            for op_name, op in self.operators.items():
                # Verifica si se cumplen las precondiciones del operador
                if all(precond in current_state for precond in op["preconditions"]):
                    new_state, has_undesired = self.apply_effects(current_state, op["effects"])
                    if has_undesired:
                        continue  # Descarta planes que generen efectos no deseados
                    new_plan = current_plan + [op_name]  # Agrega la acción al plan
                    queue.append((list(new_state), new_plan))  # Agrega el nuevo estado a la cola

        return None  # Devuelve None si no se encuentra un plan válido

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Instancia del planificador condicional
    planner = ConditionalPlanner()

    # Estado inicial y objetivo
    estado_inicial = ["CajaEnA", "RobotEnA", "SueloMojado"]
    objetivo = ["CajaEnB"]

    # Genera el plan
    plan = planner.plan(objetivo, estado_inicial)

    # Muestra los resultados
    print("\n--- Planificación Condicional Mejorada ---")
    print("Estado inicial:", estado_inicial)
    print("Objetivo:", objetivo)
    print("Plan generado:", plan)  # Debería ser: ['SecarSuelo', 'MoverCaja']