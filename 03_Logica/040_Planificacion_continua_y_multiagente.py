# Importa la librería `time` para manejar pausas en la ejecución del programa
import time
# Importa la función `random` para generar números aleatorios (usado para simular probabilidades de éxito)
from random import random
# Importa `deque` para manejar colas de acciones y `defaultdict` para manejar estructuras de datos por defecto
from collections import deque, defaultdict

# Clase principal que implementa la planificación multiagente
class MultiAgentPlanner:
    def __init__(self):
        """
        Constructor de la clase MultiAgentPlanner.
        Inicializa el estado compartido, los agentes, las ubicaciones disponibles y las plantillas de operadores.
        """
        # Estado compartido entre todos los agentes (representa el mundo)
        self.shared_state = set()
        # Diccionario para almacenar información de cada agente
        self.agents = {}
        # Lista de ubicaciones disponibles en el entorno
        self.locations = ["A", "B", "C"]
        # Plantillas de operadores que definen las acciones posibles para los agentes
        self.operator_templates = {
            "Mover": {  # Acción para mover un robot de una ubicación a otra
                "preconditions": ["RobotEn{from_loc}", "Libre{to_loc}"],  # Condiciones necesarias para ejecutar la acción
                "effects": {  # Efectos de la acción en el estado compartido
                    "add": ["RobotEn{to_loc}", "Libre{from_loc}"],  # Elementos que se añaden al estado
                    "remove": ["RobotEn{from_loc}", "Libre{to_loc}"]  # Elementos que se eliminan del estado
                },
                "duration": 2,  # Tiempo necesario para completar la acción
                "success_rate": 0.9  # Probabilidad de éxito de la acción
            },
            "Cargar": {  # Acción para cargar un paquete
                "preconditions": ["RobotEn{loc}", "PaqueteEn{loc}"],  # El robot debe estar en la misma ubicación que el paquete
                "effects": {
                    "add": ["Cargando{agent}"],  # El robot comienza a cargar el paquete
                    "remove": ["PaqueteEn{loc}"]  # El paquete ya no está en la ubicación
                },
                "duration": 3,  # Tiempo necesario para cargar
                "success_rate": 0.85  # Probabilidad de éxito
            },
            "Descargar": {  # Acción para descargar un paquete
                "preconditions": ["RobotEn{loc}", "Cargando{agent}"],  # El robot debe estar en la ubicación objetivo y cargando un paquete
                "effects": {
                    "add": ["PaqueteEn{loc}"],  # El paquete se coloca en la ubicación
                    "remove": ["Cargando{agent}"]  # El robot ya no está cargando
                },
                "duration": 2,  # Tiempo necesario para descargar
                "success_rate": 0.95  # Probabilidad de éxito
            }
        }

    def add_agent(self, agent_id, initial_location):
        """
        Añade un nuevo agente al sistema con su ubicación inicial.
        :param agent_id: Identificador único del agente.
        :param initial_location: Ubicación inicial del agente.
        """
        # Crea un nuevo agente con su estado inicial
        self.agents[agent_id] = {
            "state": {"location": initial_location, "carrying": False},  # Estado inicial del agente
            "plan": deque(),  # Cola de acciones (plan) del agente
            "current_action": None,  # Acción actual en ejecución
            "action_progress": 0,  # Progreso de la acción actual
            "goal": None  # Meta del agente (si aplica)
        }
        # Actualiza el estado compartido con la ubicación inicial del agente
        self.shared_state.add(f"RobotEn{initial_location}")
        self.shared_state.add(f"Libre{initial_location}")

    def update_world_state(self, package_locations):
        """
        Actualiza el estado compartido con las ubicaciones de los paquetes.
        :param package_locations: Lista de ubicaciones donde hay paquetes.
        """
        # Elimina cualquier información previa sobre paquetes en el estado compartido
        for loc in self.locations:
            self.shared_state.discard(f"PaqueteEn{loc}")
        # Añade las nuevas ubicaciones de los paquetes al estado compartido
        for loc in package_locations:
            self.shared_state.add(f"PaqueteEn{loc}")

    def generate_operator(self, template, agent_id, **params):
        """
        Genera un operador (acción) basado en una plantilla y parámetros específicos.
        :param template: Nombre de la plantilla de acción.
        :param agent_id: Identificador del agente que ejecutará la acción.
        :param params: Parámetros específicos para personalizar la acción.
        :return: Diccionario que representa la acción generada.
        """
        # Añade el identificador del agente a los parámetros
        params["agent"] = agent_id
        # Genera la acción con precondiciones, efectos, duración y probabilidad de éxito
        op = {
            "name": template,
            "preconditions": [p.format(**params) for p in self.operator_templates[template]["preconditions"]],
            "effects": {
                "add": [e.format(**params) for e in self.operator_templates[template]["effects"]["add"]],
                "remove": [e.format(**params) for e in self.operator_templates[template]["effects"]["remove"]]
            },
            "duration": self.operator_templates[template]["duration"],
            "success_rate": self.operator_templates[template]["success_rate"]
        }
        return op

    def check_preconditions(self, preconditions, state):
        """
        Verifica si todas las precondiciones están satisfechas en el estado actual.
        :param preconditions: Lista de precondiciones.
        :param state: Estado actual del mundo.
        :return: True si todas las precondiciones están satisfechas, False en caso contrario.
        """
        return all(p in state for p in preconditions)

    def plan_for_agent(self, agent_id, goal_location):
        """
        Genera un plan para que un agente alcance una ubicación objetivo.
        :param agent_id: Identificador del agente.
        :param goal_location: Ubicación objetivo.
        """
        agent = self.agents[agent_id]
        agent["plan"] = deque()  # Reinicia el plan actual

        current_loc = agent["state"]["location"]
        carrying = agent["state"]["carrying"]

        if not carrying:
            # Si el agente no está cargando un paquete, añade la acción de cargar
            op_cargar = self.generate_operator("Cargar", agent_id, loc=current_loc)
            agent["plan"].append(op_cargar)

        if current_loc != goal_location:
            # Si el agente no está en la ubicación objetivo, añade la acción de mover
            op_mover = self.generate_operator("Mover", agent_id, from_loc=current_loc, to_loc=goal_location)
            agent["plan"].append(op_mover)

        # Añade la acción de descargar el paquete
        op_descargar = self.generate_operator("Descargar", agent_id, loc=goal_location)
        agent["plan"].append(op_descargar)

    def execute_step(self):
        """
        Ejecuta un paso del plan para todos los agentes.
        Actualiza el estado compartido y el estado de los agentes según las acciones realizadas.
        """
        print("\n=== Paso de Ejecución ===")
        action_results = {}

        for agent_id, agent in self.agents.items():
            if agent["current_action"] is None and agent["plan"]:
                # Si no hay acción en curso, toma la siguiente del plan
                agent["current_action"] = agent["plan"].popleft()
                agent["action_progress"] = 0
                print(f"{agent_id} inicia: {agent['current_action']['name']}")

            if agent["current_action"]:
                # Progreso en la acción actual
                agent["action_progress"] += 1
                if agent["action_progress"] >= agent["current_action"]["duration"]:
                    # Verifica si la acción se completa con éxito
                    success = random() <= agent["current_action"]["success_rate"]
                    action_results[agent_id] = (agent["current_action"], success)
                    agent["current_action"] = None

        for agent_id, (action, success) in action_results.items():
            if success:
                # Si la acción fue exitosa, actualiza el estado compartido
                print(f"{agent_id} completó con éxito: {action['name']}")
                self.shared_state.update(action["effects"]["add"])
                self.shared_state.difference_update(action["effects"]["remove"])
                
                # Actualiza el estado del agente según la acción realizada
                if action["name"] == "Mover":
                    for effect in action["effects"]["add"]:
                        if "RobotEn" in effect:
                            new_loc = effect.replace("RobotEn", "")
                            self.agents[agent_id]["state"]["location"] = new_loc
                elif action["name"] == "Cargar":
                    self.agents[agent_id]["state"]["carrying"] = True
                elif action["name"] == "Descargar":
                    self.agents[agent_id]["state"]["carrying"] = False
            else:
                # Si la acción falla, se replantea el plan
                print(f"{agent_id} falló en: {action['name']} - replanning")
                current_loc = self.agents[agent_id]["state"]["location"]
                self.plan_for_agent(agent_id, "C")

        # Imprime el estado global y el estado de cada agente
        print("\nEstado global:", self.shared_state)
        for agent_id, agent in self.agents.items():
            status = f"{agent_id} en {agent['state']['location']}"
            if agent["state"]["carrying"]:
                status += " (lleva paquete)"
            print(f"{status}, Plan: {len(agent['plan'])} acciones pendientes")

# Ejecución principal
if __name__ == "__main__":
    """
    Punto de entrada principal del programa.
    Configura el entorno, añade agentes, genera planes y ejecuta la simulación.
    """
    print("=== SIMULACIÓN MULTIAGENTE MEJORADA ===")
    planner = MultiAgentPlanner()

    # Añade dos agentes con ubicaciones iniciales
    planner.add_agent("Robot1", "A")
    planner.add_agent("Robot2", "B")
    
    # Actualiza el estado inicial del mundo con paquetes en A y B
    planner.update_world_state(["A", "B"])

    # Genera planes para que ambos agentes lleven paquetes a C
    planner.plan_for_agent("Robot1", "C")
    planner.plan_for_agent("Robot2", "C")

    # Ejecuta 10 ciclos de simulación
    for step in range(10):  # 10 ciclos
        print(f"\n--- Ciclo {step + 1} ---")
        planner.execute_step()
        time.sleep(0.3)
