# Importación de tipos para anotaciones y estructuras de datos
from typing import List, Set, Dict, Tuple, Optional  # Ayuda a definir tipos de datos para mayor claridad
from collections import deque  # Cola de doble extremo, útil para implementar algoritmos de búsqueda como BFS

# Clase principal que implementa el algoritmo GraphPlan para planificación basada en grafos
class GraphPlan:
    # Clase interna que representa una acción en el problema de planificación
    class Action:
        def __init__(self, name: str, pre: Set[str], add: Set[str], delete: Set[str]):
            """
            Inicializa una acción con su nombre, precondiciones, efectos positivos (add) y efectos negativos (delete).
            :param name: Nombre de la acción (por ejemplo, "comprar_boleto").
            :param pre: Conjunto de precondiciones necesarias para ejecutar la acción.
            :param add: Conjunto de efectos positivos que se añaden al estado tras ejecutar la acción.
            :param delete: Conjunto de efectos negativos que se eliminan del estado tras ejecutar la acción.
            """
            self.name = name  # Nombre descriptivo de la acción
            self.pre = pre  # Precondiciones necesarias para que la acción sea válida
            self.add = add  # Efectos positivos que se añaden al estado
            self.delete = delete  # Efectos negativos que se eliminan del estado

    def __init__(self, actions: List[Action], initial: Set[str], goal: Set[str]):
        """
        Inicializa el planificador con las acciones, el estado inicial y el estado objetivo.
        :param actions: Lista de acciones disponibles para el planificador.
        :param initial: Estado inicial representado como un conjunto de hechos.
        :param goal: Estado objetivo representado como un conjunto de hechos.
        """
        self.actions = actions  # Lista de acciones disponibles
        self.initial = initial  # Estado inicial del problema
        self.goal = goal  # Estado objetivo que se desea alcanzar

    def plan(self) -> Optional[List[str]]:
        """
        Implementa el algoritmo de búsqueda en amplitud (BFS) para encontrar un plan que cumpla con el estado objetivo.
        :return: Lista de nombres de acciones que forman el plan, o None si no se encuentra un plan.
        """
        # Cola para la búsqueda en amplitud, inicializada con el estado inicial y un plan vacío
        queue = deque()
        queue.append((self.initial.copy(), []))  # Cada elemento es una tupla (estado_actual, plan_actual)
        
        # Conjunto de estados visitados para evitar ciclos y redundancias
        visited = set()
        visited.add(frozenset(self.initial))  # Convertimos el estado inicial en un conjunto inmutable

        # Bucle principal de búsqueda
        while queue:
            # Extraer el estado actual y el plan actual de la cola
            current_state, current_plan = queue.popleft()

            # Verificar si el estado actual cumple con el estado objetivo
            if self.goal.issubset(current_state):  # Si todos los hechos del objetivo están en el estado actual
                return current_plan  # Retornar el plan actual como solución

            # Generar todas las acciones aplicables al estado actual
            applicable_actions = []
            for action in self.actions:
                if action.pre.issubset(current_state):  # Verificar si las precondiciones de la acción se cumplen
                    applicable_actions.append(action)

            # Probar todas las combinaciones posibles de acciones aplicables
            for i in range(1, len(applicable_actions) + 1):  # Probar combinaciones de 1 a n acciones
                from itertools import combinations  # Importación local para generar combinaciones
                for action_group in combinations(applicable_actions, i):
                    # Verificar que las acciones en el grupo no sean conflictivas
                    conflict = False
                    for a1, a2 in combinations(action_group, 2):  # Comparar pares de acciones
                        # Verificar conflictos: si una acción elimina una precondición de otra
                        if a1.delete & a2.pre or a2.delete & a1.pre:
                            conflict = True
                            break
                    if conflict:  # Si hay conflicto, pasar a la siguiente combinación
                        continue

                    # Aplicar las acciones al estado actual para generar un nuevo estado
                    new_state = current_state.copy()  # Copiar el estado actual
                    action_names = []  # Lista para almacenar los nombres de las acciones aplicadas
                    for action in action_group:
                        new_state -= action.delete  # Eliminar efectos negativos
                        new_state.update(action.add)  # Añadir efectos positivos
                        action_names.append(action.name)  # Registrar el nombre de la acción

                    # Verificar si el nuevo estado ya fue visitado
                    state_key = frozenset(new_state)  # Convertir el estado en un conjunto inmutable
                    if state_key not in visited:  # Si el estado no ha sido visitado
                        visited.add(state_key)  # Marcar el estado como visitado
                        queue.append((new_state, current_plan + action_names))  # Añadir a la cola

        # Si no se encuentra un plan, devolver None
        return None

# --------------------------------------------
# Ejemplo de uso del planificador
# --------------------------------------------

if __name__ == "__main__":
    # Definición de acciones con sus precondiciones, efectos positivos y negativos
    actions = [
        GraphPlan.Action("comprar_boleto", {"dinero", "ciudad_origen"}, {"boleto"}, {"dinero"}),  # Comprar un boleto requiere dinero y estar en la ciudad de origen
        GraphPlan.Action("ir_a_estacion", {"ciudad_origen"}, {"en_estacion"}, set()),  # Ir a la estación requiere estar en la ciudad de origen
        GraphPlan.Action("abordar_tren", {"boleto", "en_estacion"}, {"en_tren"}, {"boleto", "en_estacion"}),  # Abordar el tren requiere un boleto y estar en la estación
        GraphPlan.Action("viajar", {"en_tren"}, {"en_destino"}, {"en_tren", "ciudad_origen"})  # Viajar requiere estar en el tren
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
    # Output esperado: ['comprar_boleto', 'ir_a_estacion', 'abordar_tren', 'viajar']