class HTNPlanner:
    """
    Clase que implementa un planificador HTN (Hierarchical Task Network).
    Permite descomponer tareas compuestas en subtareas primitivas y generar un plan.
    """
    def __init__(self):
        # Definición de operadores (tareas primitivas) con sus precondiciones y efectos
        self.operators = {
            "CortarPan": {
                "preconditions": ["TienePan"],  # Requiere tener pan
                "effects": ["PanCortado", "¬TienePan"]  # Produce pan cortado y elimina el pan inicial
            },
            "AgregarJamon": {
                "preconditions": ["PanCortado", "TieneJamon"],  # Requiere pan cortado y jamón
                "effects": ["JamonAgregado", "¬TieneJamon"]  # Agrega jamón y elimina el jamón inicial
            },
            "AgregarQueso": {
                "preconditions": ["PanCortado", "TieneQueso"],  # Requiere pan cortado y queso
                "effects": ["QuesoAgregado", "¬TieneQueso"]  # Agrega queso y elimina el queso inicial
            },
            "EnsamblarSandwich": {
                "preconditions": ["JamonAgregado", "QuesoAgregado"],  # Requiere jamón y queso agregados
                "effects": ["SandwichListo"]  # Produce un sándwich listo
            },
            "Servir": {
                "preconditions": ["SandwichListo"],  # Requiere un sándwich listo
                "effects": ["ComidaServida"]  # Sirve la comida
            }
        }

        # Definición de métodos (tareas compuestas) con sus subtareas
        self.methods = {
            "PrepararSandwich": [
                {"subtasks": ["CortarPan", "AgregarIngredientes", "Servir"]}  # Descomposición en subtareas
            ],
            "AgregarIngredientes": [
                {"subtasks": ["AgregarJamon", "AgregarQueso", "EnsamblarSandwich"]}  # Subtareas para agregar ingredientes
            ]
        }

    def apply_effects(self, state, effects):
        """
        Actualiza el estado con los efectos de una acción.
        :param state: Conjunto de condiciones actuales.
        :param effects: Lista de efectos de la acción.
        :return: Nuevo estado actualizado.
        """
        new_state = set(state)  # Copia del estado actual
        for effect in effects:
            if effect.startswith("¬"):  # Si el efecto es negativo (¬), elimina la condición
                new_state.discard(effect[1:])
            else:  # Si el efecto es positivo, agrega la condición
                new_state.add(effect)
        return new_state

    def plan(self, task, state):
        """
        Genera un plan HTN para la tarea dada.
        :param task: Tarea a realizar (puede ser primitiva o compuesta).
        :param state: Estado inicial.
        :return: Plan generado (lista de acciones) y estado final.
        """
        if task in self.operators:  # Si la tarea es primitiva
            op = self.operators[task]
            # Verifica si se cumplen las precondiciones
            if all(precond in state for precond in op["preconditions"]):
                # Aplica los efectos y devuelve el plan con la tarea
                return [task], self.apply_effects(state, op["effects"])
            else:
                return None, state  # Precondiciones no cumplidas

        elif task in self.methods:  # Si la tarea es compuesta
            for method in self.methods[task]:  # Prueba cada método disponible
                current_state = set(state)  # Copia del estado actual
                full_plan = []  # Plan acumulado
                for subtask in method["subtasks"]:  # Itera sobre las subtareas
                    subplan, current_state = self.plan(subtask, current_state)
                    if subplan is None:  # Si alguna subtarea falla, prueba otro método
                        break
                    full_plan.extend(subplan)  # Agrega las subtareas al plan
                else:
                    return full_plan, current_state  # Éxito: devuelve el plan completo y el estado final
            return None, state  # Ningún método funcionó

        else:
            raise ValueError(f"Error: Tarea '{task}' no definida.")  # Error si la tarea no está definida

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Instancia del planificador HTN
    planner = HTNPlanner()
    # Estado inicial con los recursos disponibles
    estado_inicial = {"TienePan", "TieneJamon", "TieneQueso"}

    # Planificación de la tarea "PrepararSandwich"
    plan, estado_final = planner.plan("PrepararSandwich", estado_inicial)

    # Resultados
    print("\n--- Planificación HTN ---")
    print("Estado inicial:", estado_inicial)  # Muestra el estado inicial
    print("Plan generado:", plan)  # Muestra el plan generado
    print("Estado final:", estado_final)  # Muestra el estado final después de ejecutar el plan