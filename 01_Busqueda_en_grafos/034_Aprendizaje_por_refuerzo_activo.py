import numpy as np
from collections import defaultdict
import random

class ActiveRLAgent:
    def __init__(self, env, alpha=0.1, gamma=0.99, epsilon=0.1):
        """
        Inicializa el agente de aprendizaje por refuerzo activo.
        
        Args:
            env: Entorno que sigue la interfaz similar a Gym.
            alpha (float): Tasa de aprendizaje (qué tan rápido se actualizan los valores Q).
            gamma (float): Factor de descuento (pondera las recompensas futuras).
            epsilon (float): Probabilidad de exploración en la política ε-greedy.
        """
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        # Q-table inicializada con ceros para cada estado y acción.
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        self.state_space_size = env.observation_space.n
        self.action_space_size = env.action_space.n
        self.visited_states = set()  # Conjunto de estados visitados.

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
            return random.randint(0, self.action_space_size - 1)
        else:
            # Explotación: elegir la acción con el mayor valor Q.
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
        current_q = self.q_table[state][action]  # Valor Q actual.
        
        # Valor máximo para el siguiente estado (si no es terminal).
        max_next_q = np.max(self.q_table[next_state]) if not done else 0
        
        # Actualización de Q-learning.
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q  # Actualizar el valor Q.

    def build_state_graph(self):
        """
        Construye un grafo de estados visitados y las transiciones entre ellos.
        
        Returns:
            dict: Grafo donde las claves son estados y los valores son nodos alcanzables.
        """
        graph = defaultdict(set)
        
        # Reconstruir el grafo basado en la Q-table.
        for state in self.visited_states:
            for action in range(self.action_space_size):
                if self.q_table[state][action] > 0:  # Solo considerar transiciones posibles.
                    # Simular la acción para obtener el siguiente estado.
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
        """
        graph = self.build_state_graph()
        
        # Usar una cola para implementar una búsqueda similar a Dijkstra.
        queue = [(start_state, [], 0)]  # (estado, camino, valor acumulado).
        visited = set()
        best_path = []
        max_value = -np.inf
        
        while queue:
            current_state, path, current_value = queue.pop(0)
            
            if current_state == goal_state:
                # Si se alcanza el estado objetivo, verificar si es el mejor camino.
                if current_value > max_value:
                    max_value = current_value
                    best_path = path
                continue
            
            if current_state in visited:
                continue
            
            visited.add(current_state)
            
            for (next_state, action, q_value) in graph.get(current_state, []):
                new_path = path + [(current_state, action)]
                new_value = current_value + q_value
                queue.append((next_state, new_path, new_value))
        
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
            3: (4, 3)   # B
        }
        
        # Definir action_space correctamente
        self.action_space = type('ActionSpace', (), {
            'n': 6,
            'sample': lambda self: random.randint(0, 5)
        })()
        
        self.observation_space = type('ObservationSpace', (), {
            'n': self.grid_size**2 * len(self.locations)**2 * 2  # Más espacio para estados
        })()
        
        self.reset()
    
    def reset(self):
        self.taxi_pos = (random.randint(0, self.grid_size-1), random.randint(0, self.grid_size-1))
        self.passenger_loc = random.randint(0, 3)
        self.destination = random.choice([loc for loc in range(4) if loc != self.passenger_loc])
        self.s = self._get_state()
        return self.s
    
    def _get_state(self):
        taxi_row, taxi_col = self.taxi_pos
        state = taxi_row * self.grid_size + taxi_col
        state += self.passenger_loc * (self.grid_size**2)
        state += self.destination * (self.grid_size**2 * len(self.locations))
        return state
    
    def step(self, action):
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
                reward = -10
        elif action == 5:  # Dejar pasajero
            if self.passenger_loc == 4 and self.taxi_pos == self.locations[self.destination]:
                self.passenger_loc = self.destination
                reward = 20
                done = True
            else:
                reward = -10
        
        self.s = self._get_state()
        return self.s, reward, done, {}

def taxi_example():
    env = TaxiEnv()
    agent = ActiveRLAgent(env, alpha=0.1, gamma=0.99, epsilon=0.1)
    
    print("Entrenando al agente...")
    episodes = 1000
    for episode in range(episodes):
        state = env.reset()
        done = False
        total_reward = 0
        
        while not done:
            action = agent.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            agent.learn(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
        
        if (episode + 1) % 100 == 0:
            print(f"Episodio {episode + 1}, Recompensa total: {total_reward}")
    
    print("\nProbando el agente entrenado...")
    state = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < 100:
        action = np.argmax(agent.q_table[state])
        next_state, reward, done, _ = env.step(action)
        total_reward += reward
        state = next_state
        steps += 1
    
    print(f"Recompensa total durante la prueba: {total_reward}")
    
    print("\nEncontrando camino óptimo usando búsqueda en grafos...")
    env.reset()
    env.taxi_pos = env.locations[env.destination]
    env.passenger_loc = env.destination
    goal_state = env._get_state()
    
    env.reset()
    start_state = env._get_state()
    
    optimal_path = agent.find_optimal_path(start_state, goal_state)
    print(f"Longitud del camino óptimo encontrado: {len(optimal_path)}")

if __name__ == "__main__":
    taxi_example()