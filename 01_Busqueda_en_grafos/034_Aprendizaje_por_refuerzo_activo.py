import numpy as np
from collections import defaultdict
import random

class ActiveRLAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Inicializa el agente de aprendizaje por refuerzo activo.

        Args:
            env: Entorno que sigue la interfaz similar a Gym.  
                 - Debe proporcionar información sobre el espacio de estados y acciones.
                 - Debe tener métodos para interactuar con el entorno (reset, step).
            alpha (float): Tasa de aprendizaje (learning rate). Controla cuánto se actualizan los valores Q en cada iteración.
                           Valores más altos dan como resultado actualizaciones más grandes, lo que puede acelerar el aprendizaje pero también
                           puede hacer que sea inestable. Valores más bajos hacen que el aprendizaje sea más lento pero más estable.
            gamma (float): Factor de descuento (discount factor). Determina la importancia de las recompensas futuras.
                           Un valor de 0 significa que el agente solo se preocupa por las recompensas inmediatas.
                           Un valor de 1 significa que el agente considera todas las recompensas futuras por igual.
            epsilon (float): Probabilidad de exploración en la política ε-greedy.  
                           Es la probabilidad de que el agente elija una acción aleatoria en lugar de la mejor acción conocida.
                           Esto ayuda al agente a explorar el entorno y potencialmente encontrar mejores caminos/políticas.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table inicializada con ceros para cada estado y acción.
        # La Q-table es un diccionario donde las claves son los estados y los valores son arrays NumPy.
        # Cada array NumPy tiene un elemento para cada acción posible en ese estado, representando el valor Q estimado
        # para tomar esa acción en ese estado.  Usamos defaultdict para crear entradas sobre la marcha.
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.state_space_size = env.observation_space.n  # Número total de estados posibles en el entorno.
        self.action_space_size = env.action_space.n  # Número total de acciones posibles en el entorno.
        self.visited_states = set()  # Conjunto para realizar un seguimiento de los estados que el agente ha visitado.
                                      # Útil para construir el grafo de estados para la búsqueda de caminos.

    def choose_action(self, state):
        """
        Selecciona una acción usando la política ε-greedy.

        Args:
            state: Estado actual.

        Returns:
            int: Acción seleccionada.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploración: elegir una acción aleatoria.
            # Genera un número aleatorio entre 0 y 1. Si es menor que epsilon, elegimos una acción aleatoria.
            return random.randint(0, self.action_space_size - 1)
        else:
            # Explotación: elegir la acción con el mayor valor Q.
            # Si el número aleatorio no es menor que epsilon, elegimos la acción que tiene el valor Q más alto
            # para el estado actual según nuestra Q-table.
            return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        """
        Actualiza la Q-table usando la regla de actualización de Q-learning.

        Args:
            state: Estado actual.
            action: Acción tomada.
            reward: Recompensa obtenida.
            next_state: Siguiente estado.
            done: Si el episodio ha terminado.
        """
        self.visited_states.add(state)  # Registrar el estado como visitado.
        current_q = self.q_table[state][action]  # Valor Q actual para el estado y la acción dados.

        # Valor máximo para el siguiente estado (si no es terminal).
        # Si el episodio no ha terminado, encontramos el valor Q máximo para el siguiente estado
        # sobre todas las acciones posibles. Si el episodio ha terminado, el valor máximo del siguiente Q es 0.
        max_next_q = np.max(self.q_table[next_state]) if not done else 0

        # Actualización de Q-learning.
        # Esta es la fórmula central de Q-learning. Actualiza nuestra estimación del valor Q
        # basado en la recompensa que recibimos y nuestra estimación del valor del siguiente estado.
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q  # Actualizar el valor Q en la tabla.

    def build_state_graph(self):
        """
        Construye un grafo de estados visitados y las transiciones entre ellos.

        Returns:
            dict: Grafo donde las claves son estados y los valores son nodos alcanzables.
                  El grafo se representa como un diccionario.  Las claves del diccionario son los estados.
                  Los valores del diccionario son conjuntos de tuplas.  Cada tupla representa una transición
                  desde el estado clave a otro estado, junto con la acción que causó la transición,
                  y el valor Q de esa transición.
        """
        graph = defaultdict(set)
        # Reconstruir el grafo basado en la Q-table.
        for state in self.visited_states:
            for action in range(self.action_space_size):
                if self.q_table[state][action] > 0:  # Solo considerar transiciones posibles.
                    # Simular la acción para obtener el siguiente estado.
                    # Necesitamos interactuar con el entorno para ver a qué estado llegaremos si tomamos
                    # una acción en un estado dado.
                    self.env.reset()
                    self.env.s = state  # Establecer el estado actual.
                    next_state, _, done, _ = self.env.step(action)
                    graph[state].add((next_state, action, self.q_table[state][action]))
        return graph

    def find_optimal_path(self, start_state, goal_state):
        """
        Encuentra el camino óptimo usando el grafo construido.  
        
        Args:
            start_state: Estado inicial.
            goal_state: Estado objetivo.

        Returns:
            list: Camino óptimo como lista de tuplas (estado, acción).
                  Devuelve una lista de tuplas, donde cada tupla contiene un estado y la acción tomada
                  para llegar a ese estado en el camino óptimo.
        """
        graph = self.build_state_graph()  # Obtiene el grafo de estados y transiciones.

        # Usar una cola para implementar una búsqueda similar a Dijkstra.
        # La cola contiene tuplas de (estado actual, camino hasta el estado actual, valor acumulado).
        queue = [(start_state, [], 0)]
        visited = set()  # Conjunto para realizar un seguimiento de los estados visitados.
        best_path = []  # Almacenará el camino óptimo.
        max_value = -np.inf  # Almacenará el valor del mejor camino encontrado hasta ahora.

        while queue:
            current_state, path, current_value = queue.pop(0)  # Obtiene el siguiente nodo de la cola.

            if current_state == goal_state:
                # Si se alcanza el estado objetivo, verificar si es el mejor camino.
                if current_value > max_value:
                    max_value = current_value
                    best_path = path
                continue  # Continúa buscando otros caminos que podrían ser mejores.

            if current_state in visited:
                continue  # Si ya hemos visitado este estado, no lo volvemos a visitar.

            visited.add(current_state)  # Marca el estado actual como visitado.

            # Itera sobre los vecinos del estado actual en el grafo.
            for next_state, action, q_value in graph.get(current_state, []):
                new_path = path + [(current_state, action)]  # Extiende el camino con el estado actual y la acción.
                new_value = current_value + q_value  # Actualiza el valor acumulado.
                queue.append((next_state, new_path, new_value))  # Añade el vecino a la cola.

        # Añadir el estado final al path
        if best_path and best_path[-1][0] != goal_state:
            best_path.append((goal_state, None))

        return best_path
class TaxiEnv:
    """Implementación simplificada del entorno Taxi"""
    def __init__(self):
        self.grid_size = 5
        self.locations = {
            0: (0, 0),  # R
            1: (0, 4),  # G
            2: (4, 0),  # Y
            3: (4, 3)  # B
        }

        # Definir action_space correctamente
        # En un entorno real de Gym, action_space sería una instancia de gym.spaces.Discrete.
        # Aquí, lo estamos simulando con un objeto simple que tiene un atributo 'n' (número de acciones)
        # y un método 'sample' para elegir una acción aleatoria.
        self.action_space = type('ActionSpace', (), {
            'n': 6,  # 6 acciones posibles en el entorno Taxi.
            'sample': lambda self: random.randint(0, 5)  # Función para muestrear una acción aleatoria.
        })()

        # Definir observation_space correctamente
        # Similar a action_space, en un entorno real de Gym, observation_space sería una instancia de
        # gym.spaces.Discrete o gym.spaces.Tuple.  Aquí, lo estamos simulando.
        # El número de estados posibles se calcula en función del tamaño de la cuadrícula y el número de ubicaciones.
        self.observation_space = type('ObservationSpace', (), {
            'n': self.grid_size**2 * len(self.locations)**2 * 2  # Más espacio para estados
        })()

        self.reset()  # Inicializa el entorno.

    def reset(self):
        """
        Reinicia el entorno a un estado inicial aleatorio.

        Returns:
            int: El estado inicial del entorno.
        """
        # El taxi comienza en una posición aleatoria en la cuadrícula.
        self.taxi_pos = (random.randint(0, self.grid_size - 1), random.randint(0, self.grid_size - 1))
        # El pasajero comienza en una de las 4 ubicaciones de recogida.
        self.passenger_loc = random.randint(0, 3)
        # El destino es una de las otras 3 ubicaciones de entrega (distinta de la ubicación de recogida).
        self.destination = random.choice([loc for loc in range(4) if loc != self.passenger_loc])
        self.s = self._get_state()  # Obtiene el ID del estado basado en la posición del taxi, la ubicación del pasajero y el destino.
        return self.s

    def _get_state(self):
        """
        Convierte la posición del taxi, la ubicación del pasajero y el destino en un único entero que representa el estado.

        Returns:
            int: El ID del estado.
        """
        taxi_row, taxi_col = self.taxi_pos
        state = taxi_row * self.grid_size + taxi_col  # Codifica la posición del taxi como un número.
        state += self.passenger_loc * (self.grid_size**2)  # Añade la ubicación del pasajero al estado.
        state += self.destination * (self.grid_size**2 * len(self.locations))  # Añade el destino al estado.
        return state

    def step(self, action):
        """
        Ejecuta una acción en el entorno.

        Args:
            action (int): La acción a realizar (0-5).

        Returns:
            tuple: (s, reward, done, {})
                   - s (int): El nuevo estado después de realizar la acción.
                   - reward (int): La recompensa obtenida al realizar la acción.
                   - done (bool): Indica si el episodio ha terminado.
                   - {} (dict): Información adicional (generalmente vacío en este entorno simplificado).
        """
        reward = -1  # Penalización por paso
        done = False

        if action == 0:  # Abajo
            new_row = min(self.taxi_pos[0] + 1, self.grid_size - 1)
            self.taxi_pos = (new_row, self.taxi_pos[1])
        elif action == 1:  # Arriba
            new_row = max(self.taxi_pos[0] - 1, 0)
            self.taxi_pos = (new_row, self.taxi_pos[1])
        elif action == 2:  # Derecha
            new_col = min(self.taxi_pos[1] + 1, self.grid_size - 1)
            self.taxi_pos = (self.taxi_pos[0], new_col)
        elif action == 3:  # Izquierda
            new_col = max(self.taxi_pos[1] - 1, 0)
            self.taxi_pos = (self.taxi_pos[0], new_col)
        elif action == 4:  # Recoger pasajero
            if self.passenger_loc < 4 and self.taxi_pos == self.locations[self.passenger_loc]:
                self.passenger_loc = 4  # En taxi
                reward = 10
            else:
                reward = -10  # Penalización por intento fallido de recogida
        elif action == 5:  # Dejar pasajero
            if self.passenger_loc == 4 and self.taxi_pos == self.locations[self.destination]:
                self.passenger_loc = self.destination
                reward = 20
                done = True  # El episodio termina con éxito
            else:
                reward = -10  # Penalización por intento fallido de entrega

        self.s = self._get_state()  # Obtiene el nuevo estado después de la acción.
        return self.s, reward, done, {}
def taxi_example():
    """
    Ejecuta un ejemplo de entrenamiento y prueba de un agente ActiveRL en el entorno Taxi.
    """
    env = TaxiEnv()  # Crea una instancia del entorno Taxi.
    agent = ActiveRLAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)  # Crea un agente ActiveRL.

    print("Entrenando al agente...")
    episodes = 1000  # Número de episodios para el entrenamiento.
    for episode in range(episodes):
        state = env.reset()  # Reinicia el entorno al comienzo de cada episodio.
        done = False
        total_reward = 0  # Recompensa acumulada para este episodio.

        while not done:
            action = agent.choose_action(state)  # Elige una acción usando la política ε-greedy.
            next_state, reward, done, _ = env.step(action)  # Da un paso en el entorno.
            agent.learn(state, action, reward, next_state, done)  # Actualiza la Q-table.
            state = next_state  # Actualiza el estado actual.
            total_reward += reward  # Acumula la recompensa.

        if (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}, Recompensa total: {total_reward}")

    print("\nProbando el agente entrenado...")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0

    while not done and steps < 100:  # Limita el número de pasos para evitar bucles infinitos.
        action = np.argmax(agent.q_table[state])  # Elige la mejor acción según la Q-table aprendida (explotación).
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1

    print(f"Recompensa total durante la prueba: {total_reward}")

    print("\nEncontrando camino óptimo usando búsqueda en grafos...")
    env.reset()
    env.taxi_pos = env.locations[env.destination]
    env.passenger_loc = env.destination
    goal_state = env._get_state()  # Estado objetivo: taxi y pasajero en el destino.

    env.reset()
    start_state = env._get_state()  # Estado inicial.

    optimal_path = agent.find_optimal_path(start_state, goal_state)  # Encuentra el camino óptimo.
    print(f"Longitud del camino óptimo encontrado: {len(optimal_path)}")


if __name__ == "__main__":
    taxi_example()  # Llama a la función de ejemplo para ejecutar el código.
