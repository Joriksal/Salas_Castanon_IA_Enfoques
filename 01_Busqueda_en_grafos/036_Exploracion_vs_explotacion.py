# Importaciones necesarias
import numpy as np
import random
import matplotlib.pyplot as plt
from collections import defaultdict, deque

# Clase del agente que implementa estrategias de exploración/explotación
class ExplorationExploitationAgent:
    def __init__(self, env, exploration_strategy='epsilon_greedy', epsilon=0.1, 
                 ucb_c=2, temperature=1.0):
        """
        Inicializa el agente con una estrategia de exploración/explotación.

        Args:
            env: Entorno (grafo o laberinto).
            exploration_strategy: Estrategia a usar ('epsilon_greedy', 'ucb', 'boltzmann').
            epsilon: Parámetro para ε-greedy.
            ucb_c: Parámetro para UCB.
            temperature: Parámetro para Boltzmann.
        """
        self.env = env
        self.exploration_strategy = exploration_strategy
        self.epsilon = epsilon
        self.ucb_c = ucb_c
        self.temperature = temperature
        
        # Inicialización de estructuras para aprendizaje
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))  # Valores Q
        self.visit_counts = defaultdict(lambda: np.zeros(env.action_space.n))  # Conteo de visitas
        self.total_visits = 0  # Total de visitas a cualquier estado
        
        # Grafo para registrar estados visitados
        self.state_graph = defaultdict(set)
    
    def select_action(self, state):
        """
        Selecciona una acción según la estrategia de exploración/explotación.

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy(state)
        elif self.exploration_strategy == 'ucb':
            return self._ucb(state)
        elif self.exploration_strategy == 'boltzmann':
            return self._boltzmann(state)
        else:
            raise ValueError("Estrategia no válida")
    
    def _epsilon_greedy(self, state):
        """
        Implementación de la estrategia ε-greedy.

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        if random.random() < self.epsilon:  # Exploración
            return random.choice(range(self.env.action_space.n))
        return np.argmax(self.q_values[state])  # Explotación
    
    def _ucb(self, state):
        """
        Implementación de Upper Confidence Bound (UCB).

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        if self.visit_counts[state].sum() == 0:  # Si no hay visitas, explora
            return random.choice(range(self.env.action_space.n))
        
        ucb_values = []
        for action in range(self.env.action_space.n):
            if self.visit_counts[state][action] == 0:  # Prioriza acciones no exploradas
                return action
            
            # Cálculo de UCB: explotación + exploración
            exploitation = self.q_values[state][action]
            exploration = self.ucb_c * np.sqrt(np.log(self.total_visits) / self.visit_counts[state][action])
            ucb_values.append(exploitation + exploration)
        
        return np.argmax(ucb_values)
    
    def _boltzmann(self, state):
        """
        Implementación de la estrategia Boltzmann (Softmax).

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        q_values = self.q_values[state]
        probabilities = np.exp(q_values / self.temperature) / np.sum(np.exp(q_values / self.temperature))
        return np.random.choice(range(self.env.action_space.n), p=probabilities)
    
    def update(self, state, action, reward, next_state):
        """
        Actualiza los valores Q y el grafo de estados.

        Args:
            state: Estado actual.
            action: Acción tomada.
            reward: Recompensa recibida.
            next_state: Estado siguiente.
        """
        # Actualizar conteos de visitas
        self.visit_counts[state][action] += 1
        self.total_visits += 1
        
        # Actualizar valores Q usando una tasa de aprendizaje adaptativa
        alpha = 1 / self.visit_counts[state][action]
        self.q_values[state][action] += alpha * (reward + np.max(self.q_values[next_state]) - self.q_values[state][action])
        
        # Actualizar grafo de estados
        self.state_graph[state].add((next_state, action, reward))
    
    def find_optimal_path(self, start_state, goal_state):
        """
        Encuentra el camino óptimo desde el estado inicial al objetivo.

        Args:
            start_state: Estado inicial.
            goal_state: Estado objetivo.

        Returns:
            Camino óptimo como una lista de (estado, acción).
        """
        # BFS modificado para buscar el camino de mayor recompensa
        queue = deque()
        queue.append((start_state, [], 0))  # (estado, camino, recompensa acumulada)
        best_path = []
        max_reward = -np.inf
        
        while queue:
            current_state, path, current_reward = queue.popleft()
            
            if current_state == goal_state:  # Si se llega al objetivo
                if current_reward > max_reward:
                    max_reward = current_reward
                    best_path = path
                continue
            
            for (next_state, action, reward) in self.state_graph.get(current_state, []):
                if next_state not in [s for s, _, _ in path]:  # Evitar ciclos
                    new_path = path + [(current_state, action)]
                    new_reward = current_reward + reward
                    queue.append((next_state, new_path, new_reward))
        
        return best_path

# Clase del entorno del laberinto
class MazeEnv:
    """Entorno de laberinto para probar las estrategias."""
    def __init__(self, size=5):
        self.size = size
        self.action_space = type('ActionSpace', (), {'n': 4})()  # 4 acciones posibles
        self.observation_space = type('ObservationSpace', (), {'n': size*size})()
        self.goal = (size-1, size-1)  # Meta en la esquina inferior derecha
        self.reset()
    
    def reset(self):
        """Reinicia el entorno a la posición inicial."""
        self.state = (0, 0)
        return self._get_state()
    
    def _get_state(self):
        """Convierte coordenadas (x, y) a un estado único."""
        x, y = self.state
        return x * self.size + y
    
    def step(self, action):
        """
        Ejecuta una acción y devuelve el resultado.

        Args:
            action: Acción a ejecutar.

        Returns:
            next_state: Estado siguiente.
            reward: Recompensa obtenida.
            done: Si se alcanzó la meta.
            info: Información adicional (vacío en este caso).
        """
        x, y = self.state
        reward = -0.1  # Penalización por movimiento
        done = False
        
        # Movimientos según la acción
        if action == 0: x = max(x-1, 0)      # Arriba
        elif action == 1: x = min(x+1, self.size-1)  # Abajo
        elif action == 2: y = max(y-1, 0)     # Izquierda
        elif action == 3: y = min(y+1, self.size-1)  # Derecha
        
        self.state = (x, y)
        
        # Recompensa por llegar a la meta
        if (x, y) == self.goal:
            reward = 10
            done = True
        
        return self._get_state(), reward, done, {}

# Función para comparar estrategias
def compare_strategies():
    """
    Compara diferentes estrategias de exploración/explotación en el entorno.
    """
    env = MazeEnv(size=5)  # Crear entorno de laberinto
    strategies = [
        ('epsilon_greedy (ε=0.1)', 'epsilon_greedy', {'epsilon': 0.1}),
        ('epsilon_greedy (ε=0.3)', 'epsilon_greedy', {'epsilon': 0.3}),
        ('UCB (c=2)', 'ucb', {'ucb_c': 2}),
        ('Boltzmann (τ=0.5)', 'boltzmann', {'temperature': 0.5}),
    ]
    
    results = {}
    
    for name, strategy, params in strategies:
        # Crear agente con la estrategia actual
        agent = ExplorationExploitationAgent(env, exploration_strategy=strategy, **params)
        rewards = []
        
        # Entrenamiento
        for episode in range(500):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = agent.select_action(state)
                next_state, reward, done, _ = env.step(action)
                agent.update(state, action, reward, next_state)
                state = next_state
                total_reward += reward
            
            rewards.append(total_reward)
        
        # Evaluación
        test_rewards = []
        for _ in range(100):
            state = env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = np.argmax(agent.q_values[state])  # Solo explotación
                next_state, reward, done, _ = env.step(action)
                state = next_state
                total_reward += reward
            
            test_rewards.append(total_reward)
        
        # Guardar resultados
        results[name] = {
            'training': rewards,
            'testing': np.mean(test_rewards)
        }
        
        # Mostrar camino óptimo encontrado
        start_state = env._get_state()
        goal_state = env.size * env.size - 1
        path = agent.find_optimal_path(start_state, goal_state)
        print(f"\n{name}:")
        print(f"Recompensa promedio en prueba: {np.mean(test_rewards):.2f}")
        print("Camino óptimo encontrado:")
        for state, action in path:
            x, y = state // env.size, state % env.size
            print(f"({x},{y})", end=" -> ")
        print("Meta")
    
    # Gráfico de comparación
    plt.figure(figsize=(12, 6))
    for name, data in results.items():
        plt.plot(np.convolve(data['training'], np.ones(10)/10, mode='valid'), label=name)
    plt.title("Comparación de estrategias de exploración/explotación")
    plt.xlabel("Episodio")
    plt.ylabel("Recompensa promedio (suavizada)")
    plt.legend()
    plt.grid()
    plt.show()

# Punto de entrada principal
if __name__ == "__main__":
    compare_strategies()