import numpy as np
# Importa la librería NumPy, que proporciona soporte para arrays multidimensionales y funciones matemáticas de alto nivel.
# Se utiliza aquí principalmente para la representación de la función de valor y para operaciones numéricas eficientes.
from collections import defaultdict
# Importa defaultdict de la librería collections. defaultdict es un diccionario que proporciona un valor por defecto
# para una clave que aún no existe, lo que simplifica la manipulación de estructuras de datos.

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
            gamma (float): Factor de descuento para recompensas futuras (0 <= gamma < 1).
                           Indica cuánto valora el agente las recompensas que recibirá en el futuro.
                           Un valor cercano a 0 hace que el agente sea más "miope", mientras que un valor cercano a 1 lo hace más "previsor".
        """
        self.maze = maze
        self.gamma = gamma
        self.rows = len(maze)  # Número de filas del laberinto.
        self.cols = len(maze[0]) if self.rows > 0 else 0  # Número de columnas.
        self.terminal_states = set()  # Conjunto para almacenar los IDs de los estados terminales (en este caso, solo el estado 'G').
        self.initialize_model()  # Llama al método para configurar las estructuras internas del agente.

    def initialize_model(self):
        """
        Inicializa las estructuras necesarias para el aprendizaje pasivo.
        Esto incluye el mapeo de estados a coordenadas y viceversa, la identificación de estados terminales e inicial,
        el almacenamiento de recompensas asociadas a cada estado y el registro de las transiciones observadas.
        """
        self.state_coords = {}  # Diccionario para mapear un ID de estado (entero único) a sus coordenadas (fila, columna) en el laberinto.
        self.coord_states = {}  # Diccionario para mapear coordenadas (fila, columna) a un ID de estado único.
        state_id = 0  # Contador para asignar un ID único a cada estado accesible (no pared) en el laberinto.

        # Itera a través de cada celda del laberinto para identificar los estados y sus propiedades.
        for i in range(self.rows):
            for j in range(self.cols):
                if self.maze[i][j] == '#':  # Si la celda es una pared, no es un estado accesible, así que se ignora.
                    continue
                self.state_coords[state_id] = (i, j)
                self.coord_states[(i, j)] = state_id
                if self.maze[i][j] == 'G':  # Si la celda contiene 'G' (Goal), se añade el ID de este estado al conjunto de estados terminales.
                    self.terminal_states.add(state_id)
                elif self.maze[i][j] == 'S':  # Si la celda contiene 'S' (Start), se guarda el ID de este estado como el estado inicial.
                    self.start_state = state_id
                state_id += 1

        self.num_states = state_id  # El número total de estados accesibles en el laberinto.
        self.rewards = np.zeros(self.num_states)  # Array NumPy para almacenar la recompensa inmediata asociada a cada estado. Inicializado con ceros.
        self.transitions = defaultdict(list)  # Diccionario donde las claves son los IDs de los estados y los valores son listas de tuplas (acción, nuevo_estado)
                                               # que representan las transiciones observadas desde ese estado. defaultdict asegura que si se accede a una clave inexistente,
                                               # se crea una entrada con una lista vacía como valor predeterminado.

        # Asigna las recompensas a cada estado basándose en el tipo de celda en el laberinto.
        for state in range(self.num_states):
            i, j = self.state_coords[state]
            if self.maze[i][j] == 'G':  # El estado objetivo tiene una recompensa de +1.
                self.rewards[state] = 1.0
            else:  # Todos los demás estados transitables ('.', 'S') tienen una pequeña penalización de -0.04.
                self.rewards[state] = -0.04

    def observe_episode(self, episode):
        """
        Observa un episodio completo de la interacción del agente con el entorno
        y registra las transiciones (estado, acción) -> nuevo_estado que ocurren en él.

        Args:
            episode (list): Lista de tuplas (estado, acción, nuevo_estado, recompensa) que representan la secuencia de eventos en un episodio.
                           Cada tupla describe un paso en el que el agente se encontraba en un 'estado', tomó una 'acción',
                           llegando a un 'nuevo_estado' y recibiendo una 'recompensa'.
        """
        for state, action, new_state, reward in episode:
            # Para cada paso del episodio, se registra la transición desde el 'estado' al 'nuevo_estado' al tomar la 'acción'.
            # La información de la recompensa ya está almacenada en self.rewards, por lo que aquí solo se guarda la transición.
            self.transitions[state].append((action, new_state))

    def estimate_value_function(self, method='policy_evaluation', policy=None, max_iter=1000, theta=1e-6):
        """
        Estima la función de valor (V(s)) para todos los estados utilizando el método especificado:
        - 'policy_evaluation': Evalúa el valor de los estados bajo una política dada, iterando hasta la convergencia.
        - 'value_iteration': Encuentra la función de valor óptima iterativamente, aplicando la ecuación de Bellman de optimalidad.

        Args:
            method (str): El método a utilizar para la estimación ('policy_evaluation' o 'value_iteration'). Por defecto es 'policy_evaluation'.
            policy (dict, optional): La política a evaluar (solo necesaria cuando method='policy_evaluation').
                                     Debe ser un diccionario donde las claves son los IDs de los estados y los valores son listas de acciones posibles en ese estado.
                                     Si no se proporciona una política para la evaluación, se asume una política aleatoria uniforme (todas las acciones tienen la misma probabilidad).
            max_iter (int): El número máximo de iteraciones a realizar en el proceso de estimación. Esto evita bucles infinitos en caso de no convergencia.
            theta (float): Un pequeño valor umbral para determinar la convergencia de los valores de los estados.
                           La iteración se detiene cuando la mayor diferencia absoluta entre los valores de los estados en dos iteraciones consecutivas es menor que theta.

        Returns:
            np.array: Un array NumPy que contiene la función de valor estimada para cada estado. El índice del array corresponde al ID del estado.
        """
        V = np.zeros(self.num_states)  # Inicializa un array para almacenar los valores de los estados, todos comenzando en 0.

        if method == 'policy_evaluation' and policy is None:
            # Si se elige la evaluación de política pero no se proporciona una política, se crea una política aleatoria uniforme.
            # Para cada estado, se asume que todas las acciones ('up', 'down', 'left', 'right') son igualmente probables.
            policy = {s: ['up', 'down', 'left', 'right'] for s in range(self.num_states)}

        for _ in range(max_iter):
            # Itera hasta alcanzar el número máximo de iteraciones.
            delta = 0  # Inicializa la máxima diferencia en los valores de los estados en esta iteración a 0.
            old_V = V.copy()  # Crea una copia de los valores de los estados de la iteración anterior para poder calcular la diferencia.

            for state in range(self.num_states):
                # Itera sobre todos los estados accesibles en el laberinto.
                if state in self.terminal_states:  # Los estados terminales tienen un valor fijo (su recompensa) y no se actualizan.
                    continue

                if method == 'policy_evaluation':
                    # Implementación del paso de evaluación de política.
                    expected_value = 0
                    possible_actions = policy[state]
                    if possible_actions:
                        for action in possible_actions:
                            # Filtra las transiciones observadas desde el estado actual para la acción específica.
                            transitions_for_action = [ns for a, ns in self.transitions[state] if a == action]
                            if transitions_for_action:
                                # Calcula la probabilidad empírica de tomar esta acción desde este estado, basada en las observaciones.
                                prob = len(transitions_for_action) / len(self.transitions[state]) if self.transitions[state] else 0
                                if prob > 0:
                                    # Calcula el estado siguiente promedio (si hay múltiples ocurrencias de esta acción, se toma el promedio de los estados resultantes).
                                    next_state = int(np.mean(transitions_for_action))
                                    # Actualiza el valor esperado sumando la recompensa inmediata del estado actual y el valor descontado del estado siguiente.
                                    expected_value += (prob / len(possible_actions)) * (self.rewards[state] + self.gamma * old_V[next_state])
                        # El nuevo valor del estado es el valor esperado calculado bajo la política.
                        V[state] = expected_value

                elif method == 'value_iteration':
                    # Implementación del paso de iteración de valor.
                    max_q_value = -np.inf
                    for action in ['up', 'down', 'left', 'right']:
                        # Itera sobre todas las acciones posibles desde el estado actual.
                        transitions_for_action = [ns for a, ns in self.transitions[state] if a == action]
                        if transitions_for_action:
                            prob = len(transitions_for_action) / len(self.transitions[state]) if self.transitions[state] else 0
                            if prob > 0:
                                next_state = int(np.mean(transitions_for_action))
                                # Calcula el valor Q (calidad de la acción): recompensa inmediata del estado actual + valor descontado del estado siguiente.
                                q_value = prob * (self.rewards[state] + self.gamma * old_V[next_state])
                                if q_value > max_q_value:
                                    max_q_value = q_value
                    # El nuevo valor del estado es el máximo valor Q encontrado sobre todas las acciones posibles.
                    V[state] = max_q_value if max_q_value > -np.inf else 0

                # Calcula la diferencia absoluta entre el valor del estado en la iteración anterior y el valor actual.
                delta = max(delta, np.abs(old_V[state] - V[state]))

            # Si la máxima diferencia entre los valores de los estados en esta iteración y la anterior es menor que el umbral theta,
            # se considera que los valores han convergido y se detiene el proceso iterativo.
            if delta < theta:
                break

        return V  # Devuelve la función de valor estimada.

    def extract_policy(self, V):
        """
        Extrae una política greedy (que elige la acción que maximiza el valor esperado del siguiente estado)
        basada en la función de valor estimada V.

        Args:
            V (np.array): La función de valor estimada para cada estado.

        Returns:
            dict: Un diccionario que mapea cada estado a la lista de acciones óptimas (aquellas que conducen al estado con el valor más alto esperado).
                  Si hay múltiples acciones que llevan a estados con el mismo valor máximo, todas esas acciones se incluyen en la lista.
        """
        policy = {}
        for state in range(self.num_states):
            if state in self.terminal_states:
                # Los estados terminales no tienen acciones asociadas.
                policy[state] = None
                continue

            best_actions = []
            max_action_value = -np.inf

            for action in ['up', 'down', 'left', 'right']:
                transitions_for_action = [ns for a, ns in self.transitions[state] if a == action]
                if transitions_for_action:
                    prob = len(transitions_for_action) / len(self.transitions[state]) if self.transitions[state] else 0
                    if prob > 0:
                        next_state = int(np.mean(transitions_for_action))
                        # Calcula el valor de tomar la 'acción' en el 'estado' actual, que es la recompensa inmediata más el valor descontado del estado siguiente.
                        action_value = prob * (self.rewards[state] + self.gamma * V[next_state])

                        if action_value > max_action_value:
                            # Si el valor de esta acción es mayor que el máximo valor encontrado hasta ahora para este estado,
                            # se actualiza el máximo valor y se establece esta acción como la única mejor acción.
                            max_action_value = action_value
                            best_actions = [action]
                        elif action_value == max_action_value:
                            # Si el valor de esta acción es igual al máximo valor encontrado, se añade esta acción a la lista de mejores acciones.
                            # Esto permite que la política sea probabilística en caso de empates en los valores de las acciones.
                            best_actions.append(action)

            policy[state] = best_actions  # Asigna la lista de mejores acciones encontradas para el estado actual a la política.

        return policy  # Devuelve la política extraída.

def generate_random_episode(maze, max_steps=100):
    """
    Genera un episodio aleatorio de la interacción del agente con el laberinto.
    El agente comienza en el estado 'S' y realiza acciones aleatorias (arriba, abajo, izquierda, derecha)
    hasta alcanzar el estado objetivo 'G' o hasta que se alcance el número máximo de pasos.

    Args:
        maze (list of lists): Representación del laberinto.
        max_steps (int): El número máximo de pasos permitidos en el episodio para evitar bucles infinitos.

    Returns:
        list: Una lista de tuplas (estado, acción, nuevo_estado, recompensa) que representan la secuencia de eventos en el episodio.
              Cada tupla describe un paso de la interacción del agente con el entorno.
    """
    # Encuentra la posición inicial 'S' en el laberinto.
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

    # Mapea las coordenadas (fila, columna) a un ID de estado único.
    for i in range(len(maze)):
        for j in range(len(maze[0])):
            if maze[i][j] != '#':
                coord_states[(i, j)] = state_id
                state_id += 1

    for _ in range(max_steps):
        i, j = current_pos
        current_state = coord_states[(i, j)]

        if maze[i][j] == 'G':  # Termina el episodio si el agente alcanza el estado objetivo.
            break

        # Define los posibles movimientos (acciones) y los cambios de coordenadas correspondientes.
        possible_actions = []
        new_positions = []
        action_deltas = {'up': (-1, 0), 'down': (1, 0), 'left': (0, -1), 'right': (0, 1)}

        for action, (di, dj) in action_deltas.items():
            new_i, new_j = i + di, j + dj
            # Verifica si la nueva posición está dentro de los límites del laberinto y no es una pared.
            if 0 <= new_i < len(maze) and 0 <= new_j < len(maze[0]) and maze[new_i][new_j] != '#':
                possible_actions.append(action)
                new_positions.append((new_i, new_j))

        if not possible_actions:  # Si no hay movimientos posibles desde la posición actual, termina el episodio.
            break

        # Elige una acción aleatoria de la lista de acciones posibles.
        action_idx = np.random.randint(len(possible_actions))
        action = possible_actions[action_idx]
        new_pos = new_positions[action_idx]
        new_state = coord_states[new_pos]

        # Determina la recompensa obtenida al llegar al nuevo estado.
        reward = -0.04
        if maze[new_pos[0]][new_pos[1]] == 'G':
            reward = 1.0

        # Añade la tupla (estado actual, acción tomada, nuevo estado, recompensa obtenida) al episodio.
        episode.append((current_state, action, new_state, reward))
        # Añade una tupla que representa el paso actual al episodio.
        # La tupla contiene:
        # - current_state: El ID del estado en el que el agente se encontraba antes de tomar la acción.
        # - action: La acción que el agente tomó desde el current_state.
        # - new_state: El ID del estado al que el agente llegó después de tomar la acción.
        # - reward: La recompensa que el agente recibió al llegar al new_state.
        current_pos = new_pos  # Actualiza la posición actual del agente a las coordenadas del nuevo estado.

    return episode  # Devuelve la lista de pasos (el episodio completo) que el agente experimentó.

def print_maze_with_values(maze, values, coord_states):
    """
    Imprime el laberinto en la consola, mostrando el valor estimado para cada estado accesible.
    Esta función ayuda a visualizar la función de valor aprendida por el agente sobre el laberinto.

    Args:
        maze (list of lists): Representación del laberinto (la misma estructura que se usó para inicializar el agente).
        values (np.array): Array NumPy que contiene los valores estimados de cada estado. El índice del array corresponde al ID del estado.
        coord_states (dict): Diccionario que mapea las coordenadas (fila, columna) a los IDs de los estados.
                             Se utiliza para encontrar el ID del estado correspondiente a cada celda del laberinto.
    """
    for i in range(len(maze)):
        # Itera sobre cada fila del laberinto.
        row_str = []  # Inicializa una lista para almacenar la representación de la fila actual.
        for j in range(len(maze[0])):
            # Itera sobre cada columna de la fila actual.
            if maze[i][j] == '#':
                # Si la celda es una pared ('#'), añade '#####' a la representación de la fila.
                # Se usan varios '#' para que visualmente ocupe más espacio y sea fácil de identificar.
                row_str.append('#####')
            else:
                # Si la celda no es una pared, es un estado accesible.
                state = coord_states[(i, j)]  # Obtiene el ID del estado correspondiente a las coordenadas (i, j).
                if maze[i][j] in ['S', 'G']:
                    # Si el estado es el estado inicial ('S') o el objetivo ('G'), añade un prefijo para identificarlo claramente.
                    prefix = maze[i][j] + ':'
                else:
                    # Para los estados transitables ('.'), añade un prefijo más corto para mantener la alineación.
                    prefix = '  :'
                row_str.append(f"{prefix}{values[state]:.2f}")  # Formatea el valor del estado (con dos decimales) y lo añade a la representación de la fila.
        print(' '.join(row_str))  # Une los elementos de la lista row_str con espacios para formar una cadena que representa la fila del laberinto y la imprime.

# Ejemplo: Laberinto 4x4
def maze_example():
    """
    Ejecuta un ejemplo de aprendizaje por refuerzo pasivo en un laberinto 4x4.
    Este ejemplo demuestra cómo se puede utilizar la clase PassiveRLAgent para aprender la función de valor
    de los estados en un entorno de laberinto basado en la observación de episodios aleatorios.
    Finalmente, extrae una política greedy basada en la función de valor aprendida.
    """
    # Definición del laberinto como una lista de listas de caracteres.
    # 'S': Estado inicial, '#': Pared, '.': Estado transitable, 'G': Estado objetivo.
    maze = [
        ['S', '.', '.', '.'],
        ['.', '#', '.', '#'],
        ['.', '.', '.', '.'],
        ['#', '.', '#', 'G']
    ]

    # Crea una instancia de la clase PassiveRLAgent, pasando el laberinto y el factor de descuento gamma.
    agent = PassiveRLAgent(maze)

    # Generar y observar un cierto número de episodios aleatorios.
    print("Generando episodios de entrenamiento...")
    num_episodes = 100  # Define el número de episodios que se generarán para la observación.
    for i in range(num_episodes):
        # Genera un episodio aleatorio de la interacción del agente con el laberinto.
        episode = generate_random_episode(maze)
        # El agente observa el episodio, lo que actualiza su registro de transiciones (self.transitions).
        agent.observe_episode(episode)
        print(f"Episodio {i+1} generado con {len(episode)} pasos.")

    # Estimar la función de valor utilizando el método de evaluación de política.
    print("\nEstimando función de valor con evaluación de política...")
    # Llama al método estimate_value_function con el método 'policy_evaluation'.
    # Como no se proporciona una política explícita, se utilizará una política aleatoria uniforme por defecto.
    V_policy = agent.estimate_value_function(method='policy_evaluation')
    # Imprime el laberinto mostrando los valores de los estados estimados por la evaluación de política.
    print_maze_with_values(maze, V_policy, agent.coord_states)

    # Extraer una política greedy basada en la función de valor estimada por la evaluación de política.
    print("\nExtrayendo política greedy basada en la evaluación de política...")
    # Llama al método extract_policy para obtener una política que siempre elige la acción que conduce al estado con el valor más alto esperado.
    policy_eval = agent.extract_policy(V_policy)

    print("\nPolítica óptima estimada (Evaluación de Política):")
    # Itera sobre todos los estados accesibles para imprimir la política aprendida.
    for state in range(agent.num_states):
        if state in agent.terminal_states:
            # Los estados terminales no tienen acciones.
            continue
        i, j = agent.state_coords[state]  # Obtiene las coordenadas del estado.
        print(f"Estado ({i},{j}): {policy_eval[state]}")  # Imprime el estado y la lista de acciones óptimas desde ese estado.

    # Estimar la función de valor utilizando el método de iteración de valor.
    print("\nEstimando función de valor con iteración de valor...")
    # Llama al método estimate_value_function con el método 'value_iteration'.
    # La iteración de valor directamente busca la función de valor óptima.
    V_value = agent.estimate_value_function(method='value_iteration')
    # Imprime el laberinto mostrando los valores de los estados estimados por la iteración de valor.
    print_maze_with_values(maze, V_value, agent.coord_states)

    # Extraer una política greedy basada en la función de valor estimada por la iteración de valor.
    print("\nExtrayendo política greedy basada en la iteración de valor...")
    # Llama al método extract_policy para obtener la política óptima basada en la función de valor óptima.
    policy_value = agent.extract_policy(V_value)

    print("\nPolítica óptima estimada (Iteración de Valor):")
    # Itera sobre todos los estados accesibles para imprimir la política óptima aprendida.
    for state in range(agent.num_states):
        if state in agent.terminal_states:
            # Los estados terminales no tienen acciones.
            continue
        i, j = agent.state_coords[state]  # Obtiene las coordenadas del estado.
        print(f"Estado ({i},{j}): {policy_value[state]}")  # Imprime el estado y la lista de acciones óptimas desde ese estado.

if __name__ == "__main__":
    # Este bloque de código se ejecuta solo cuando el script se llama directamente (no cuando se importa como un módulo).
    maze_example()  # Llama a la función principal para ejecutar el ejemplo del laberinto.