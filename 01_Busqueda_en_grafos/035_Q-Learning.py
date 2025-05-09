import numpy as np  # Importa la librería NumPy para realizar operaciones numéricas eficientes.
                    # NumPy proporciona soporte para arrays multidimensionales y funciones matemáticas de alto nivel.
import random  # Importa la librería random para generar números aleatorios.
                    # Se utiliza para la exploración en el algoritmo Q-Learning, permitiendo al agente elegir acciones aleatorias.
from collections import defaultdict  # Importa la clase defaultdict del módulo collections.
                    # defaultdict es un tipo de diccionario que asigna un valor predeterminado a las claves que aún no han sido inicializadas.
                    # En este caso, se utiliza para inicializar la tabla Q.

# Clase que implementa el agente Q-Learning
class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, exploration_rate=0.1):
        """
        Inicializa el agente Q-Learning.
        
        Args:
            env: Entorno (laberinto). El entorno define el espacio de estados, el espacio de acciones y la dinámica de cómo el agente se mueve a través del laberinto.
            learning_rate (float): Tasa de aprendizaje (alpha). Determina cuánto se actualizan los valores Q en cada paso de aprendizaje.
            discount_factor (float): Factor de descuento (gamma). Determina la importancia de las recompensas futuras en comparación con las recompensas inmediatas.
            exploration_rate (float): Probabilidad de exploración (epsilon). Controla la frecuencia con la que el agente elige una acción aleatoria en lugar de la mejor acción conocida.
        """
        self.env = env  # Entorno en el que el agente interactúa
        self.alpha = learning_rate  # Tasa de aprendizaje
        self.gamma = discount_factor  # Factor de descuento para recompensas futuras
        self.epsilon = exploration_rate  # Probabilidad de explorar en lugar de explotar
        # Tabla Q inicializada con valores por defecto (0 para todas las acciones)
        # La tabla Q es un diccionario donde las claves son los estados y los valores son arrays de numpy.
        # Cada array de numpy representa los valores Q para cada acción posible desde ese estado.
        self.q_table = defaultdict(lambda: np.zeros(env.action_space.n))
        
    def choose_action(self, state):
        """
        Selecciona una acción usando la estrategia ε-greedy.
        
        Args:
            state: Estado actual. Un entero que representa la ubicación actual del agente en el laberinto.
            
        Returns:
            int: Acción seleccionada (0: arriba, 1: abajo, 2: izquierda, 3: derecha). Un entero que indica qué acción debe tomar el agente.
        """
        if random.uniform(0, 1) < self.epsilon:
            # Exploración: elige una acción aleatoria
            # Con probabilidad epsilon, el agente elige una acción al azar. Esto ayuda al agente a descubrir nuevas rutas y evitar quedarse atascado en un camino subóptimo.
            return random.choice(range(self.env.action_space.n))
        else:
            # Explotación: elige la mejor acción según la tabla Q
            # Con probabilidad 1-epsilon, el agente elige la acción que tiene el valor Q más alto en la tabla Q para el estado actual.
            return np.argmax(self.q_table[state])
    
    def update_q_table(self, state, action, reward, next_state):
        """
        Actualiza la Q-table usando la ecuación de Q-Learning.
        
        Args:
            state: Estado actual.
            action: Acción tomada.
            reward: Recompensa obtenida después de tomar la acción en el estado actual.
            next_state: Siguiente estado al que llegó el agente después de tomar la acción.
        """
        # Encuentra la mejor acción posible en el siguiente estado
        # Esto es parte de la ecuación de Q-Learning. Encontramos la acción que el agente tomaría en el siguiente estado
        # si estuviera actuando de forma puramente codiciosa (es decir, eligiendo siempre la mejor acción).
        best_next_action = np.argmax(self.q_table[next_state])
        # Calcula el valor objetivo (TD target)
        # El valor objetivo es la estimación de cuál debería ser el valor Q para el estado y la acción actuales.
        # Es la recompensa inmediata más el valor descontado del mejor valor Q posible en el siguiente estado.
        td_target = reward + self.gamma * self.q_table[next_state][best_next_action]
        # Calcula el error temporal (TD error)
        # El error temporal es la diferencia entre el valor Q actual y el valor objetivo.
        td_error = td_target - self.q_table[state][action]
        # Actualiza el valor Q para el estado y acción actuales
        # Actualizamos el valor Q en la tabla Q moviéndolo en la dirección del error temporal, escalado por la tasa de aprendizaje.
        self.q_table[state][action] += self.alpha * td_error
    
    def train(self, episodes=1000):
        """
        Entrena al agente en el entorno.
        
        Args:
            episodes (int): Número de episodios de entrenamiento. Un episodio es una ejecución completa del agente a través del laberinto,
                            desde el inicio hasta la meta.
        """
        for episode in range(episodes):
            state = self.env.reset()  # Reinicia el entorno al inicio de cada episodio
            done = False  # Marca para indicar si el episodio ha terminado
            
            while not done:
                # Selecciona una acción
                action = self.choose_action(state)
                # Ejecuta la acción y obtiene el siguiente estado, recompensa y si terminó
                # El agente da un paso en el entorno, moviéndose a un nuevo estado y recibiendo una recompensa.
                # El entorno también indica si el episodio ha terminado (por ejemplo, si el agente llega a la meta).
                next_state, reward, done, _ = self.env.step(action)
                # Actualiza la tabla Q
                self.update_q_table(state, action, reward, next_state)
                # Avanza al siguiente estado
                state = next_state
            
            # Imprime información cada 100 episodios
            if (episode + 1) % 100 == 0:
                # Imprimimos la recompensa media de los valores Q para dar una idea del progreso del aprendizaje.
                print(f"Episodio {episode + 1}, Recompensa media: {np.mean(list(self.q_table.values())):.2f}")
    
    def find_optimal_path(self, start_state, goal_state):
        """
        Encuentra el camino óptimo usando la Q-table.
        
        Args:
            start_state: Estado inicial. El estado desde donde el agente comienza a buscar un camino.
            goal_state: Estado objetivo. El estado al que el agente intenta llegar.
            
        Returns:
            list: Lista de estados en el camino óptimo. Una lista ordenada de los estados que el agente debe atravesar
                  para llegar del estado inicial al estado objetivo, según lo determinado por la tabla Q aprendida.
        """
        path = []  # Lista para almacenar el camino
        current_state = start_state  # Comienza desde el estado inicial
        visited = set()  # Conjunto para evitar ciclos
        
        while current_state != goal_state and current_state not in visited:
            visited.add(current_state)  # Marca el estado como visitado
            # Selecciona la mejor acción según la tabla Q
            # El agente elige la acción que tiene el valor Q más alto en la tabla Q para el estado actual.
            best_action = np.argmax(self.q_table[current_state])
            # Ejecuta la acción para obtener el siguiente estado
            # El agente da un paso en el entorno, moviéndose al siguiente estado basado en la mejor acción.
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
        # Esto define las acciones que el agente puede tomar en el entorno.
        self.action_space = type('ActionSpace', (), {'n': 4})()
        # Espacio de observación: número total de celdas en el laberinto
        # Esto define el número de estados distintos en el entorno.
        self.observation_space = type('ObservationSpace', (), {'n': size * size})()
        self.goal = (size - 1, size - 1)  # Meta en la esquina inferior derecha
        self.reset()  # Inicializa el entorno
    
    def reset(self):
        """Reinicia el entorno."""
        self.state = (0, 0)  # Estado inicial en la esquina superior izquierda
        return self._get_state()
    
    def _get_state(self):
        """Convierte la posición (x, y) en un estado único."""
        x, y = self.state  # Obtiene las coordenadas x e y del agente.
        return x * self.size + y  # Convierte las coordenadas a un único número entero, que representa el estado.
    
    def step(self, action):
        """
        Ejecuta una acción en el entorno.
        
        Args:
            action (int): 0=arriba, 1=abajo, 2=izquierda, 3=derecha. Un entero que indica qué acción tomar.
            
        Returns:
            tuple: (next_state, reward, done, info)
            next_state (int): El siguiente estado al que llega el agente después de tomar la acción.
            reward (float): La recompensa obtenida por el agente al realizar la acción.
            done (bool): Un valor booleano que indica si el episodio ha terminado (es decir, si el agente ha llegado a la meta).
            info (dict): Un diccionario que contiene información de diagnóstico (generalmente vacío en este entorno simplificado).
        """
        x, y = self.state  # Obtiene la posición actual
        done = False
        reward = -0.1  # Penalización por movimiento para incentivar caminos cortos
        
        # Actualiza la posición según la acción
        if action == 0:  # Arriba
            x = max(x - 1, 0)  # Evita salirse del límite superior
        elif action == 1:  # Abajo
            x = min(x + 1, self.size - 1)  # Evita salirse del límite inferior
        elif action == 2:  # Izquierda
            y = max(y - 1, 0)  # Evita salirse del límite izquierdo
        elif action == 3:  # Derecha
            y = min(y + 1, self.size - 1)  # Evita salirse del límite derecho
        
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
