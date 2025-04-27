from typing import List, Dict, Set, Tuple, Optional

# ----------------------------- Estructuras de Datos -----------------------------
class Action:
    """Clase que representa una acción con precondiciones, efectos positivos y efectos negativos."""
    def __init__(self, name: str, preconditions: Set[str], add_effects: Set[str], del_effects: Set[str]):
        self.name = name  # Nombre de la acción
        self.preconditions = preconditions  # Conjunto de precondiciones necesarias para ejecutar la acción
        self.add_effects = add_effects  # Conjunto de efectos que se añaden al estado tras ejecutar la acción
        self.del_effects = del_effects  # Conjunto de efectos que se eliminan del estado tras ejecutar la acción

class PartialOrderPlanner:
    """Clase que implementa un Planificador de Orden Parcial (POP)."""
    def __init__(self, actions: List[Action], initial_state: Set[str], goal_state: Set[str]):
        self.actions = actions  # Lista de acciones disponibles
        self.initial_state = initial_state  # Estado inicial del problema
        self.goal_state = goal_state  # Estado objetivo que se desea alcanzar

    def find_open_preconditions(self, plan: List[Action]) -> List[Tuple[int, str]]:
        """
        Encuentra precondiciones no satisfechas en el plan actual.
        Devuelve una lista de tuplas (índice de la acción, precondición no satisfecha).
        """
        open_preconds = []
        for i, action in enumerate(plan):
            for precond in action.preconditions:
                # Verifica si la precondición no está satisfecha por ninguna acción previa
                if not any(precond in a.add_effects for a in plan[:i]):
                    open_preconds.append((i, precond))
        return open_preconds

    def resolve_threats(self, plan: List[Action], causal_links: List[Tuple[int, int, str]]) -> bool:
        """
        Detecta y resuelve amenazas a los enlaces causales.
        Una amenaza ocurre si una acción intermedia elimina un efecto necesario para otra acción.
        """
        for (i, j, p) in causal_links:  # Recorre los enlaces causales
            for k, action in enumerate(plan):
                if p in action.del_effects and i < k < j:  # Si una acción intermedia amenaza el enlace
                    # Resolver amenaza añadiendo restricciones de orden (k < i) o (j < k)
                    if not (k < i or j < k):
                        return False  # No se puede resolver la amenaza
        return True

    def plan(self) -> Optional[List[str]]:
        """
        Genera un plan parcialmente ordenado que satisface el estado objetivo.
        Devuelve una lista de nombres de acciones en el orden necesario.
        """
        # Inicializar el plan con acciones ficticias de inicio y fin
        start = Action("START", set(), self.initial_state, set())  # Acción inicial
        end = Action("END", self.goal_state, set(), set())  # Acción final
        plan = [start, end]  # Plan inicial con solo START y END
        causal_links = []  # Lista de enlaces causales (acción origen, acción destino, precondición satisfecha)
        ordering_constraints = [(0, 1)]  # Restricción inicial: START < END

        # Bucle principal: iterar hasta que todas las precondiciones estén satisfechas
        while True:
            open_preconds = self.find_open_preconditions(plan)  # Buscar precondiciones abiertas
            if not open_preconds:
                break  # Si no hay precondiciones abiertas, el plan está completo

            # Seleccionar la primera precondición abierta
            (action_idx, precond) = open_preconds[0]

            # Buscar acciones que puedan satisfacer la precondición
            possible_actions = [
                (i, a) for i, a in enumerate(self.actions) 
                if precond in a.add_effects
            ]
            if not possible_actions:
                return None  # No hay plan posible si no se puede satisfacer una precondición

            # Elegir la primera acción posible (se puede optimizar con heurísticas)
            new_action_idx, new_action = possible_actions[0]
            plan.insert(action_idx, new_action)  # Insertar la acción en el plan

            # Añadir un enlace causal y restricciones de orden
            causal_links.append((new_action_idx, action_idx, precond))
            ordering_constraints.append((new_action_idx, action_idx))

            # Resolver amenazas a los enlaces causales
            if not self.resolve_threats(plan, causal_links):
                return None  # Si no se pueden resolver las amenazas, no hay plan posible

        # Extraer y devolver los nombres de las acciones (excluyendo START y END)
        return [a.name for a in plan[1:-1]]

# ----------------------------- Ejemplo: Transporte de Paquetes -----------------------------
if __name__ == "__main__":
    # Definir acciones disponibles en el dominio del problema
    actions = [
        Action(
            name="cargar",  # Acción para cargar el paquete en el camión
            preconditions={"paquete_en_A", "camion_en_A"},  # Requiere que el paquete y el camión estén en A
            add_effects={"paquete_en_camion"},  # El paquete estará en el camión
            del_effects={"paquete_en_A"}  # El paquete ya no estará en A
        ),
        Action(
            name="mover_A_B",  # Acción para mover el camión de A a B
            preconditions={"camion_en_A"},  # Requiere que el camión esté en A
            add_effects={"camion_en_B"},  # El camión estará en B
            del_effects={"camion_en_A"}  # El camión ya no estará en A
        ),
        Action(
            name="descargar",  # Acción para descargar el paquete en B
            preconditions={"paquete_en_camion", "camion_en_B"},  # Requiere que el paquete esté en el camión y el camión en B
            add_effects={"paquete_en_B"},  # El paquete estará en B
            del_effects={"paquete_en_camion"}  # El paquete ya no estará en el camión
        )
    ]

    # Definir el estado inicial y el estado objetivo
    initial_state = {"paquete_en_A", "camion_en_A"}  # El paquete y el camión están en A
    goal_state = {"paquete_en_B"}  # El paquete debe estar en B

    # Crear el planificador y generar el plan
    planner = PartialOrderPlanner(actions, initial_state, goal_state)
    plan = planner.plan()  # Generar el plan
    print("Plan de Orden Parcial:", plan)  # Mostrar el plan generado