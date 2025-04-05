import numpy as np
import random
from collections import defaultdict

# Clase que implementa el agente Q-Learning
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            env: Entorno (laberinto).
            learning_rate (float): Tasa de aprendizaje (alpha).
            discount_factor (float): Factor de descuento (gamma).
            exploration_rate (float): Probabilidad de exploración (epsilon).
        """
        self.env = env  # Entorno en el que el agente interactúa
        self.alpha = learning_rate  # Tasa de aprendizaje
        self.gamma = discount_factor  # Factor de descuento para recompensas futuras
        self.epsilon = exploration_rate  # Probabilidad de explorar en lugar de explotar
        # Tabla Q inicializada con valores por defecto (0 para todas las acciones)
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def choose_action(self, state):
        """
        Selecciona una acción usando la estrategia ε-greedy.
        
        Args:
            state: Estado actual.
            
        Returns:
            int: Acción seleccionada (0: arriba, 1: abajo, 2: izquierda, 3: derecha).
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploración: elige una acción aleatoria
            return random.choice(range(self.env.action_space.n))
        else:
            # Explotación: elige la mejor acción según la tabla Q
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Actualiza la Q-table usando la ecuación de Q-Learning.
        
        Args:
            state: Estado actual.
            action: Acción tomada.
            reward: Recompensa obtenida.
            next_state: Siguiente estado.
        """
        # Encuentra la mejor acción posible en el siguiente estado
        best_next_action = np.argmax(self.q_table[next_state])
        # Calcula el valor objetivo (TD target)
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        # Calcula el error temporal (TD error)
        td_error = td_target - self.q_table[state][action]
        # Actualiza el valor Q para el estado y acción actuales
        self.q_table[state][action] += self.alpha * td_error
    
    def train(self, episodes=1000):
        """
        Entrena al agente en el entorno.
        
        Args:
            episodes (int): Número de episodios de entrenamiento.
        """
        for episode in range(episodes):
            state = self.env.reset()  # Reinicia el entorno al inicio de cada episodio
            done = False
            
            while not done:
                # Selecciona una acción
                action = self.choose_action(state)
                # Ejecuta la acción y obtiene el siguiente estado, recompensa y si terminó
                next_state, reward, done, _ = self.env.step(action)
                # Actualiza la tabla Q
                self.update_q_table(state, action, reward, next_state)
                # Avanza al siguiente estado
                state = next_state
            
            # Imprime información cada 100 episodios
            if (episode + 1) % 100 == 0:
                print(f"Episodio {episode + 1}, Recompensa media: {np.mean(list(self.q_table.values())):.2f}")
    
    def find_optimal_path(self, start_state, goal_state):
        """
        Encuentra el camino óptimo usando la Q-table.
        
        Args:
            start_state: Estado inicial.
            goal_state: Estado objetivo.
            
        Returns:
            list: Lista de estados en el camino óptimo.
        """
        path = []  # Lista para almacenar el camino
        current_state = start_state  # Comienza desde el estado inicial
        visited = set()  # Conjunto para evitar ciclos
        
        while current_state != goal_state and current_state not in visited:
            visited.add(current_state)  # Marca el estado como visitado
            # Selecciona la mejor acción según la tabla Q
            best_action = np.argmax(self.q_table[current_state])
            # Ejecuta la acción para obtener el siguiente estado
            next_state, _, _, _ = self.env.step(best_action)
            path.append(current_state)  # Agrega el estado actual al camino
            current_state = next_state  # Avanza al siguiente estado
        
        # Si se alcanza el objetivo, lo agrega al camino
        if current_state == goal_state:
            path.append(goal_state)
        
        return path

# Clase que define el entorno del laberinto
class MazeEnv:
    """Entorno de laberinto simple."""
    def __init__(self, size=5):
        self.size = size  # Tamaño del laberinto (size x size)
        # Espacio de acciones: 4 posibles (arriba, abajo, izquierda, derecha)
        self.action_space = type('ActionSpace', (), {'n': 4})()
        # Espacio de observación: número total de celdas en el laberinto
        self.observation_space = type('ObservationSpace', (), {'n': size * size})()
        self.goal = (size - 1, size - 1)  # Meta en la esquina inferior derecha
        self.reset()  # Inicializa el entorno
    
    def reset(self):
        """Reinicia el entorno."""
        self.state = (0, 0)  # Estado inicial en la esquina superior izquierda
        return self._get_state()
    
    def _get_state(self):
        """Convierte la posición (x, y) en un estado único."""
        x, y = self.state
        return x * self.size + y  # Convierte coordenadas a un índice único
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action (int): 0=arriba, 1=abajo, 2=izquierda, 3=derecha.
            
        Returns:
            tuple: (next_state, reward, done, info)
        """
        x, y = self.state  # Obtiene la posición actual
        done = False
        reward = -0.1  # Penalización por movimiento para incentivar caminos cortos
        
        # Actualiza la posición según la acción
        if action == 0:  # Arriba
            x = max(x - 1, 0)
        elif action == 1:  # Abajo
            x = min(x + 1, self.size - 1)
        elif action == 2:  # Izquierda
            y = max(y - 1, 0)
        elif action == 3:  # Derecha
            y = min(y + 1, self.size - 1)
        
        self.state = (x, y)  # Actualiza el estado
        
        # Si alcanza la meta, asigna recompensa y marca como terminado
        if (x, y) == self.goal:
            reward = 1.0  # Recompensa por llegar a la meta
            done = True
        
        return self._get_state(), reward, done, {}

# Función principal
def main():
    # Crea el entorno del laberinto de tamaño 5x5
    env = MazeEnv(size=5)
    # Crea el agente Q-Learning con parámetros específicos
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1)
    
    print("Entrenando al agente Q-Learning...")
    agent.train(episodes=1000)  # Entrena al agente durante 1000 episodios
    
    print("\nEncontrando el camino óptimo...")
    start_state = env._get_state()  # Estado inicial
    goal_state = env.size * env.size - 1  # Estado objetivo (última celda)
    optimal_path = agent.find_optimal_path(start_state, goal_state)  # Encuentra el camino óptimo
    
    # Imprime el camino óptimo
    print(f"Camino óptimo desde (0, 0) hasta ({env.size - 1}, {env.size - 1}):")
    for state in optimal_path:
        x = state // env.size
        y = state % env.size
        print(f"→ ({x}, {y})", end=" ")
    print("\n¡Meta alcanzada!")

# Ejecuta el programa principal
if __name__ == "__main__":
    main()