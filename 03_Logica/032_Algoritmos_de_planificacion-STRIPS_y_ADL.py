from typing import Set, List, Dict, Optional, Union  # Para definir tipos de datos como conjuntos, listas, diccionarios, valores opcionales y uniones de tipos, mejorando la legibilidad y el control de tipos.

# ----------------------------- Definiciones Básicas -----------------------------
class StripsAction:
    """Representa una acción STRIPS clásica (sin efectos condicionales)."""
    def __init__(self, name: str, preconditions: Set[str], add_effects: Set[str], del_effects: Set[str]):
        self.name = name  # Nombre de la acción
        self.preconditions = preconditions  # Condiciones necesarias para ejecutar la acción
        self.add_effects = add_effects  # Efectos que se añaden al estado tras ejecutar la acción
        self.del_effects = del_effects  # Efectos que se eliminan del estado tras ejecutar la acción

class AdlAction:
    """Representa una acción ADL que incluye efectos condicionales."""
    def __init__(self, name: str, preconditions: Set[str], effects: List[Dict]):
        self.name = name  # Nombre de la acción
        self.preconditions = preconditions  # Condiciones necesarias para ejecutar la acción
        self.effects = effects  # Lista de efectos condicionales (cada uno con condición, efectos a añadir y eliminar)

class Planner:
    """Planificador que soporta acciones STRIPS y ADL."""
    def __init__(self, actions: List[Union[StripsAction, AdlAction]]):
        self.actions = actions  # Lista de acciones disponibles en el dominio

    def find_plan(self, initial_state: Set[str], goal_state: Set[str]) -> Optional[List[str]]:
        """
        Encuentra un plan para alcanzar el estado objetivo desde el estado inicial.
        :param initial_state: Conjunto de hechos que representan el estado inicial.
        :param goal_state: Conjunto de hechos que representan el estado objetivo.
        :return: Lista de nombres de acciones que forman el plan, o None si no hay plan.
        """
        current_state = set(initial_state)  # Copia del estado actual
        plan = []  # Lista para almacenar el plan encontrado
        visited_states = []  # Lista para evitar ciclos en los estados visitados

        # Bucle hasta que el estado actual cumpla con el estado objetivo
        while not goal_state.issubset(current_state):
            if current_state in visited_states:
                return None  # Evita ciclos devolviendo None si el estado ya fue visitado
            visited_states.append(set(current_state))

            # Buscar acciones aplicables en el estado actual
            applicable_actions = []
            for action in self.actions:
                if isinstance(action, StripsAction):  # Para acciones STRIPS
                    if action.preconditions.issubset(current_state):
                        applicable_actions.append(action)
                elif isinstance(action, AdlAction):  # Para acciones ADL
                    if all(p in current_state for p in action.preconditions):
                        applicable_actions.append(action)

            if not applicable_actions:
                return None  # Si no hay acciones aplicables, no hay plan posible

            # Elegir la primera acción aplicable (se puede mejorar con heurísticas)
            chosen_action = applicable_actions[0]
            plan.append(chosen_action.name)  # Añadir el nombre de la acción al plan

            # Aplicar los efectos de la acción elegida
            if isinstance(chosen_action, StripsAction):
                current_state -= chosen_action.del_effects  # Eliminar efectos negativos
                current_state |= chosen_action.add_effects  # Añadir efectos positivos
            elif isinstance(chosen_action, AdlAction):
                for effect in chosen_action.effects:
                    # Verificar si se cumplen las condiciones del efecto
                    if all(c in current_state for c in effect.get("condition", set())):
                        current_state -= effect.get("del", set())  # Eliminar efectos negativos
                        current_state |= effect.get("add", set())  # Añadir efectos positivos

        return plan  # Devolver el plan encontrado

# ----------------------------- Ejemplo: Dominio de Logística -----------------------------
# Definición de una acción STRIPS (sin condiciones)
load_package = StripsAction(
    name="load_package",  # Nombre de la acción
    preconditions={"package_at_A", "truck_at_A", "hand_empty"},  # Condiciones necesarias
    add_effects={"package_in_truck"},  # Efectos positivos
    del_effects={"package_at_A", "hand_empty"}  # Efectos negativos
)

# Definición de una acción ADL (con efectos condicionales)
move_truck = AdlAction(
    name="move_truck_A_B",  # Nombre de la acción
    preconditions={"truck_at_A"},  # Condiciones necesarias
    effects=[
        {  # Efecto si el camión no está averiado
            "condition": {"truck_not_broken"},  # Condición para este efecto
            "add": {"truck_at_B", "package_at_B"},  # Efectos positivos
            "del": {"truck_at_A", "package_at_A"}  # Efectos negativos
        },
        {  # Efecto por defecto (si el camión está averiado)
            "condition": set(),  # Sin condición específica
            "add": {"truck_at_B"},  # Efectos positivos
            "del": {"truck_at_A"}  # Efectos negativos
        }
    ]
)

# Definición del estado inicial y el estado objetivo
initial_state = {"package_at_A", "truck_at_A", "hand_empty", "truck_not_broken"}  # Estado inicial
goal_state = {"package_at_B"}  # Estado objetivo

# Creación del planificador y búsqueda del plan
planner = Planner([load_package, move_truck])  # Crear el planificador con las acciones definidas
plan = planner.find_plan(initial_state, goal_state)  # Buscar el plan
print("Plan encontrado:", plan)  # Output esperado: ['load_package', 'move_truck_A_B']