class HTNPlanner:
    """
    Clase que implementa un planificador HTN (Hierarchical Task Network).
    Este planificador permite descomponer tareas compuestas en subtareas primitivas
    y generar un plan secuencial para alcanzar un objetivo dado.
    """

    def __init__(self):
        """
        Constructor de la clase HTNPlanner.
        Define los operadores (tareas primitivas) y los métodos (tareas compuestas).
        """
        # Definición de operadores (tareas primitivas) con sus precondiciones y efectos
        # Cada operador representa una acción básica que puede ejecutarse directamente.
        self.operators = {
            "CortarPan": {
                "preconditions": ["TienePan"],  # Condiciones necesarias para ejecutar esta acción
                "effects": ["PanCortado", "¬TienePan"]  # Efectos que produce la acción
            },
            "AgregarJamon": {
                "preconditions": ["PanCortado", "TieneJamon"],  # Requiere pan cortado y jamón disponible
                "effects": ["JamonAgregado", "¬TieneJamon"]  # Agrega jamón y elimina el recurso inicial
            },
            "AgregarQueso": {
                "preconditions": ["PanCortado", "TieneQueso"],  # Requiere pan cortado y queso disponible
                "effects": ["QuesoAgregado", "¬TieneQueso"]  # Agrega queso y elimina el recurso inicial
            },
            "EnsamblarSandwich": {
                "preconditions": ["JamonAgregado", "QuesoAgregado"],  # Requiere que el jamón y el queso estén agregados
                "effects": ["SandwichListo"]  # Produce un sándwich listo para servir
            },
            "Servir": {
                "preconditions": ["SandwichListo"],  # Requiere que el sándwich esté listo
                "effects": ["ComidaServida"]  # Sirve la comida como resultado final
            }
        }

        # Definición de métodos (tareas compuestas) con sus subtareas
        # Los métodos descomponen tareas complejas en subtareas más simples.
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
        Actualiza el estado actual aplicando los efectos de una acción.
        :param state: Conjunto de condiciones actuales (estado inicial).
        :param effects: Lista de efectos de la acción (positivos o negativos).
        :return: Nuevo estado actualizado después de aplicar los efectos.
        """
        # Crea una copia del estado actual para evitar modificar el original
        new_state = set(state)
        for effect in effects:
            if effect.startswith("¬"):  # Si el efecto es negativo (¬), elimina la condición del estado
                new_state.discard(effect[1:])  # Elimina la condición sin el símbolo "¬"
            else:  # Si el efecto es positivo, agrega la condición al estado
                new_state.add(effect)
        return new_state  # Devuelve el estado actualizado

    def plan(self, task, state):
        """
        Genera un plan HTN para alcanzar la tarea dada desde el estado inicial.
        :param task: Tarea a realizar (puede ser primitiva o compuesta).
        :param state: Estado inicial que describe las condiciones actuales.
        :return: Plan generado (lista de acciones) y estado final después de ejecutar el plan.
        """
        if task in self.operators:  # Si la tarea es primitiva (definida en los operadores)
            op = self.operators[task]  # Obtiene los detalles del operador
            # Verifica si todas las precondiciones de la tarea se cumplen en el estado actual
            if all(precond in state for precond in op["preconditions"]):
                # Si las precondiciones se cumplen, aplica los efectos y devuelve el plan con la tarea
                return [task], self.apply_effects(state, op["effects"])
            else:
                # Si las precondiciones no se cumplen, no se puede ejecutar la tarea
                return None, state

        elif task in self.methods:  # Si la tarea es compuesta (definida en los métodos)
            # Itera sobre los métodos disponibles para esta tarea
            for method in self.methods[task]:
                current_state = set(state)  # Copia del estado actual para probar el método
                full_plan = []  # Lista para acumular las acciones del plan
                # Intenta ejecutar cada subtarea definida en el método
                for subtask in method["subtasks"]:
                    subplan, current_state = self.plan(subtask, current_state)  # Planifica la subtarea
                    if subplan is None:  # Si alguna subtarea falla, prueba otro método
                        break
                    full_plan.extend(subplan)  # Agrega las acciones de la subtarea al plan completo
                else:
                    # Si todas las subtareas se ejecutan con éxito, devuelve el plan completo y el estado final
                    return full_plan, current_state
            # Si ningún método funciona, devuelve None y el estado original
            return None, state

        else:
            # Si la tarea no está definida ni como operador ni como método, lanza un error
            raise ValueError(f"Error: Tarea '{task}' no definida.")

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Instancia del planificador HTN
    planner = HTNPlanner()

    # Estado inicial con los recursos disponibles
    estado_inicial = {"TienePan", "TieneJamon", "TieneQueso"}  # Condiciones iniciales

    # Planificación de la tarea "PrepararSandwich"
    plan, estado_final = planner.plan("PrepararSandwich", estado_inicial)

    # Resultados
    print("\n--- Planificación HTN ---")
    print("Estado inicial:", estado_inicial)  # Muestra el estado inicial antes de planificar
    print("Plan generado:", plan)  # Muestra el plan generado como una lista de acciones
    print("Estado final:", estado_final)  # Muestra el estado final después de ejecutar el plan