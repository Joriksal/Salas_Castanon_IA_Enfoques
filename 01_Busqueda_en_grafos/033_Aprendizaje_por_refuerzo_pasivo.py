import numpy as np
from collections import defaultdict, deque

class PassiveRLAgent:
    def __init__(self, maze, gamma=0.9):
        """
        Inicializa el agente de aprendizaje por refuerzo pasivo.
        
        Args:
            maze (list of lists): Representación matricial del laberinto
                - 'S': Estado inicial
                - 'G': Estado objetivo (terminal con recompensa +1)
                - '#': Pared (estado no accesible)
                - '.': Estado transitable (recompensa -0.04)
            gamma (float): Factor de descuento para recompensas futuras.
        """
        self.maze = maze
        self.gamma = gamma
        self.rows = len(maze)  # Número de filas del laberinto.
        self.cols = len(maze[0]) if self.rows > 0 else 0  # Número de columnas.
        self.terminal_states = set()  # Conjunto de estados terminales.
        self.initialize_model()  # Inicializar el modelo del agente.
        
    def initialize_model(self):
        """
        Inicializa las estructuras necesarias para el aprendizaje pasivo.
        """
        # Mapeo de estados a coordenadas y viceversa.
        self.state_coords = {}  # Mapea un ID de estado a coordenadas (i, j).
        self.coord_states = {}  # Mapea coordenadas (i, j) a un ID de estado.
        state_id = 0  # Contador para asignar IDs únicos a los estados.
        
        # Identificar estados iniciales, terminales y transitables.
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == '#':  # Ignorar paredes.
                    continue
                self.state_coords[state_id] = (i, j)
                self.coord_states[(i, j)] = state_id
                if self.maze[i][j] == 'G':  # Estado objetivo.
                    self.terminal_states.add(state_id)
                elif self.maze[i][j] == 'S':  # Estado inicial.
                    self.start_state = state_id
                state_id += 1
        
        self.num_states = state_id  # Total de estados accesibles.
        self.rewards = np.zeros(self.num_states)  # Recompensas para cada estado.
        self.transitions = defaultdict(list)  # Transiciones observadas.
        
        # Inicializar recompensas para cada estado.
        for state in range(self.num_states):
            i, j = self.state_coords[state]
            if self.maze[i][j] == 'G':  # Recompensa positiva en el objetivo.
                self.rewards[state] = 1.0
            else:  # Penalización pequeña en estados transitables.
                self.rewards[state] = -0.04
    
    def observe_episode(self, episode):
        """
        Observa un episodio completo y actualiza el modelo con las transiciones observadas.
        
        Args:
            episode (list): Lista de tuplas (estado, acción, nuevo_estado, recompensa).
        """
        for state, action, new_state, reward in episode:
            self.transitions[state].append((action, new_state))  # Registrar transición.
    
    def estimate_value_function(self, method='policy_evaluation', policy=None, max_iter=1000, theta=1e-6):
        """
        Estima la función de valor usando el método especificado.
        
        Args:
            method (str): 'policy_evaluation' o 'value_iteration'.
            policy (dict): Política a evaluar (solo para policy_evaluation).
            max_iter (int): Máximo número de iteraciones.
            theta (float): Umbral de convergencia.
            
        Returns:
            np.array: Función de valor estimada.
        """
        V = np.zeros(self.num_states)  # Inicializar valores de los estados en 0.
        
        if method == 'policy_evaluation' and policy is None:
            # Si no se proporciona una política, usar una política aleatoria uniforme.
            policy = {s: ['up', 'down', 'left', 'right'] for s in range(self.num_states)}
        
        for _ in range(max_iter):
            delta = 0  # Diferencia máxima entre valores antiguos y nuevos.
            old_V = V.copy()  # Copiar los valores actuales.
            
            for state in range(self.num_states):
                if state in self.terminal_states:  # Saltar estados terminales.
                    continue
                
                if method == 'policy_evaluation':
                    # Evaluación de política: calcular el valor esperado bajo la política.
                    total = 0
                    possible_actions = policy[state]
                    for action in possible_actions:
                        # Transiciones observadas para esta acción.
                        transitions_for_action = [ns for a, ns in self.transitions[state] if a == action]
                        if not transitions_for_action:
                            continue
                        
                        # Probabilidad empírica de la acción.
                        prob = len(transitions_for_action) / len(self.transitions[state])
                        # Estado siguiente (promedio si hay múltiples transiciones).
                        next_state = int(np.mean(transitions_for_action))
                        total += prob * (self.rewards[state] + self.gamma * old_V[next_state])
                    
                    if possible_actions:
                        V[state] = total / len(possible_actions)
                
                elif method == 'value_iteration':
                    # Iteración de valor: encontrar la mejor acción posible.
                    max_value = -np.inf
                    for action in ['up', 'down', 'left', 'right']:
                        transitions_for_action = [ns for a, ns in self.transitions[state] if a == action]
                        if not transitions_for_action:
                            continue
                        
                        # Probabilidad empírica de la acción.
                        prob = len(transitions_for_action) / len(self.transitions[state])
                        next_state = int(np.mean(transitions_for_action))
                        current_value = prob * (self.rewards[state] + self.gamma * old_V[next_state])
                        if current_value > max_value:
                            max_value = current_value
                    
                    if max_value > -np.inf:
                        V[state] = max_value
            
            delta = np.max(np.abs(old_V - V))  # Calcular el cambio máximo.
            if delta < theta:  # Verificar criterio de convergencia.
                break
        
        return V  # Devolver la función de valor estimada.
    
    def extract_policy(self, V):
        """
        Extrae una política greedy basada en la función de valor estimada.
        
        Args:
            V (np.array): Función de valor estimada.
            
        Returns:
            dict: Política óptima para cada estado.
        """
        policy = {}
        
        for state in range(self.num_states):
            if state in self.terminal_states:  # No hay acciones en estados terminales.
                policy[state] = None
                continue
            
            best_actions = []  # Lista de mejores acciones.
            max_value = -np.inf  # Valor máximo inicial.
            
            for action in ['up', 'down', 'left', 'right']:
                transitions_for_action = [ns for a, ns in self.transitions[state] if a == action]
                if not transitions_for_action:
                    continue
                
                # Probabilidad empírica de la acción.
                prob = len(transitions_for_action) / len(self.transitions[state])
                next_state = int(np.mean(transitions_for_action))
                action_value = prob * (self.rewards[state] + self.gamma * V[next_state])
                
                if action_value > max_value:
                    max_value = action_value
                    best_actions = [action]
                elif action_value == max_value:
                    best_actions.append(action)
            
            policy[state] = best_actions  # Asignar las mejores acciones al estado.
        
        return policy  # Devolver la política óptima.

def generate_random_episode(maze, max_steps=100):
    """
    Genera un episodio aleatorio en el laberinto.
    
    Args:
        maze (list of lists): Representación del laberinto.
        max_steps (int): Máximo número de pasos antes de terminar el episodio.
        
    Returns:
        list: Episodio como lista de tuplas (state, action, new_state, reward).
    """
    # Encuentra el estado inicial.
    start_pos = None
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] == 'S':
                start_pos = (i, j)
                break
        if start_pos:
            break
    
    if not start_pos:
        raise ValueError("No se encontró estado inicial 'S' en el laberinto.")
    
    episode = []
    current_pos = start_pos
    coord_states = {}
    state_id = 0
    
    # Mapeo de coordenadas a estados.
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] != '#':
                coord_states[(i, j)] = state_id
                state_id += 1
    
    for _ in range(max_steps):
        i, j = current_pos
        current_state = coord_states[(i, j)]
        
        if maze[i][j] == 'G':  # Terminar si se llega al objetivo.
            break
        
        # Posibles movimientos (arriba, abajo, izquierda, derecha).
        possible_actions = []
        new_positions = []
        
        for action, (di, dj) in zip(['up', 'down', 'left', 'right'], 
                                    [(-1,0), (1,0), (0,-1), (0,1)]):
            new_i, new_j = i + di, j + dj
            if 0 <= new_i < len(maze) and 0 <= new_j < len(maze[0]) and maze[new_i][new_j] != '#':
                possible_actions.append(action)
                new_positions.append((new_i, new_j))
        
        if not possible_actions:  # Si no hay movimientos posibles, terminar.
            break
        
        # Elegir acción aleatoria.
        action_idx = np.random.randint(len(possible_actions))
        action = possible_actions[action_idx]
        new_pos = new_positions[action_idx]
        new_state = coord_states[new_pos]
        
        # Determinar recompensa.
        reward = -0.04
        if maze[new_pos[0]][new_pos[1]] == 'G':
            reward = 1.0
        
        episode.append((current_state, action, new_state, reward))
        current_pos = new_pos  # Actualizar posición actual.
    
    return episode  # Devolver el episodio generado.

def print_maze_with_values(maze, values, coord_states):
    """
    Imprime el laberinto con los valores estimados para cada estado.
    
    Args:
        maze (list of lists): Representación del laberinto.
        values (np.array): Valores estimados para cada estado.
        coord_states (dict): Mapeo de coordenadas a estados.
    """
    for i in range(len(maze)):
        row_str = []
        for j in range(len(maze[0])):
            if maze[i][j] == '#':  # Paredes.
                row_str.append('#####')
            else:
                state = coord_states[(i, j)]
                if maze[i][j] in ['S', 'G']:  # Estado inicial o objetivo.
                    prefix = maze[i][j] + ':'
                else:
                    prefix = '  :'
                row_str.append(f"{prefix}{values[state]:.2f}")
        print(' '.join(row_str))

# Ejemplo: Laberinto 4x4
def maze_example():
    """
    Ejecuta un ejemplo de aprendizaje por refuerzo pasivo en un laberinto 4x4.
    """
    maze = [
        ['S', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '.'],
        ['#', '.', '#', 'G']
    ]
    
    agent = PassiveRLAgent(maze)
    
    # Generar y observar 100 episodios aleatorios.
    print("Generando episodios de entrenamiento...")
    for _ in range(100):
        episode = generate_random_episode(maze)
        agent.observe_episode(episode)
    
    # Estimar función de valor con evaluación de política.
    print("\nEstimando función de valor con evaluación de política...")
    V_policy = agent.estimate_value_function(method='policy_evaluation')
    print_maze_with_values(maze, V_policy, agent.coord_states)
    
    # Extraer política greedy.
    print("\nExtrayendo política greedy...")
    policy = agent.extract_policy(V_policy)
    
    print("\nPolítica óptima estimada:")
    for state in range(agent.num_states):
        if state in agent.terminal_states:
            continue
        i, j = agent.state_coords[state]
        print(f"Estado ({i},{j}): {policy[state]}")
    
    # Estimar función de valor con iteración de valor.
    print("\nEstimando función de valor con iteración de valor...")
    V_value = agent.estimate_value_function(method='value_iteration')
    print_maze_with_values(maze, V_value, agent.coord_states)

if __name__ == "__main__":
    maze_example()