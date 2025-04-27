from typing import List, Set, Dict, Tuple, Optional
from collections import deque

class GraphPlan:
    # Clase interna que representa una acción en el problema de planificación
    class Action:
        def __init__(self, name: str, pre: Set[str], add: Set[str], delete: Set[str]):
            """
            Inicializa una acción con su nombre, precondiciones, efectos positivos (add) y efectos negativos (delete).
            :param name: Nombre de la acción.
            :param pre: Conjunto de precondiciones necesarias para ejecutar la acción.
            :param add: Conjunto de efectos positivos que se añaden al estado tras ejecutar la acción.
            :param delete: Conjunto de efectos negativos que se eliminan del estado tras ejecutar la acción.
            """
            self.name = name
            self.pre = pre
            self.add = add
            self.delete = delete

    def __init__(self, actions: List[Action], initial: Set[str], goal: Set[str]):
        """
        Inicializa el planificador con las acciones, el estado inicial y el estado objetivo.
        :param actions: Lista de acciones disponibles.
        :param initial: Estado inicial representado como un conjunto de hechos.
        :param goal: Estado objetivo representado como un conjunto de hechos.
        """
        self.actions = actions
        self.initial = initial
        self.goal = goal

    def plan(self) -> Optional[List[str]]:
        """
        Implementa el algoritmo de búsqueda en amplitud (BFS) para encontrar un plan que cumpla con el estado objetivo.
        :return: Lista de nombres de acciones que forman el plan, o None si no se encuentra un plan.
        """
        # Cola para la búsqueda en amplitud, inicializada con el estado inicial y un plan vacío
        queue = deque()
        queue.append((self.initial.copy(), []))
        
        # Conjunto de estados visitados para evitar ciclos
        visited = set()
        visited.add(frozenset(self.initial))

        while queue:
            # Extraer el estado actual y el plan actual de la cola
            current_state, current_plan = queue.popleft()

            # Verificar si el estado actual cumple con el estado objetivo
            if self.goal.issubset(current_state):
                return current_plan

            # Generar todas las acciones aplicables al estado actual
            applicable_actions = []
            for action in self.actions:
                if action.pre.issubset(current_state):  # Verificar precondiciones
                    applicable_actions.append(action)

            # Probar todas las combinaciones posibles de acciones aplicables
            for i in range(1, len(applicable_actions) + 1):
                from itertools import combinations
                for action_group in combinations(applicable_actions, i):
                    # Verificar que las acciones en el grupo no sean conflictivas
                    conflict = False
                    for a1, a2 in combinations(action_group, 2):
                        if a1.delete & a2.pre or a2.delete & a1.pre:  # Conflicto entre acciones
                            conflict = True
                            break
                    if conflict:
                        continue

                    # Aplicar las acciones al estado actual para generar un nuevo estado
                    new_state = current_state.copy()
                    action_names = []
                    for action in action_group:
                        new_state -= action.delete  # Eliminar efectos negativos
                        new_state.update(action.add)  # Añadir efectos positivos
                        action_names.append(action.name)

                    # Verificar si el nuevo estado ya fue visitado
                    state_key = frozenset(new_state)
                    if state_key not in visited:
                        visited.add(state_key)
                        queue.append((new_state, current_plan + action_names))

        # Si no se encuentra un plan, devolver None
        return None

# --------------------------------------------
# Ejemplo 
# --------------------------------------------

if __name__ == "__main__":
    # Definición de acciones con sus precondiciones, efectos positivos y negativos
    actions = [
        GraphPlan.Action("comprar_boleto", {"dinero", "ciudad_origen"}, {"boleto"}, {"dinero"}),
        GraphPlan.Action("ir_a_estacion", {"ciudad_origen"}, {"en_estacion"}, set()),
        GraphPlan.Action("abordar_tren", {"boleto", "en_estacion"}, {"en_tren"}, {"boleto", "en_estacion"}),
        GraphPlan.Action("viajar", {"en_tren"}, {"en_destino"}, {"en_tren", "ciudad_origen"})
    ]
    
    # Estado inicial: el viajero tiene dinero y está en la ciudad de origen
    initial_state = {"dinero", "ciudad_origen"}
    
    # Estado objetivo: el viajero debe estar en el destino
    goal_state = {"en_destino"}

    # Crear una instancia del planificador y generar el plan
    planner = GraphPlan(actions, initial_state, goal_state)
    travel_plan = planner.plan()
    
    # Imprimir el plan generado
    print("\nPlan de viaje CORRECTO:", travel_plan)
    # Output garantizado: ['comprar_boleto', 'ir_a_estacion', 'abordar_tren', 'viajar']