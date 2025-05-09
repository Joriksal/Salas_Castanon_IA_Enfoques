# Importaciones necesarias
import numpy as np  # Importa la librería NumPy para operaciones numéricas eficientes.
import random  # Importa la librería random para generar números aleatorios.
import matplotlib.pyplot as plt  # Importa Matplotlib para visualización de datos.
from collections import defaultdict, deque  # Importa estructuras de datos útiles.

# Clase del agente que implementa estrategias de exploración/explotación
class ExplorationExploitationAgent:
    def __init__(self, env, exploration_strategy='epsilon_greedy', epsilon=0.1,
                 ucb_c=2, temperature=1.0):
        """
        Inicializa el agente con una estrategia de exploración/explotación.

        Args:
            env: Entorno (grafo o laberinto). El entorno define el espacio de estados, las acciones posibles y las reglas de transición.
            exploration_strategy: Estrategia a usar ('epsilon_greedy', 'ucb', 'boltzmann').  Determina cómo el agente balancea la exploración de nuevas acciones y la explotación de las acciones conocidas.
            epsilon: Parámetro para ε-greedy. Controla la probabilidad de que el agente elija una acción aleatoria.
            ucb_c: Parámetro para UCB.  Controla la influencia de la incertidumbre en la selección de acciones.
            temperature: Parámetro para Boltzmann.  Controla la aleatoriedad en la selección de acciones basada en probabilidades.
        """
        self.env = env  # Guarda el entorno del agente.
        self.exploration_strategy = exploration_strategy  # Guarda la estrategia de exploración seleccionada.
        self.epsilon = epsilon  # Parámetro para la exploración ε-greedy.
        self.ucb_c = ucb_c  # Parámetro para la exploración UCB.
        self.temperature = temperature  # Parámetro para la exploración de Boltzmann.

        # Inicialización de estructuras para aprendizaje
        self.q_values = defaultdict(lambda: np.zeros(env.action_space.n))  # Tabla de valores Q inicializada.
                                        #  Un diccionario que mapea estados a un array de valores Q para cada acción.
        self.visit_counts = defaultdict(lambda: np.zeros(env.action_space.n))  # Contador de visitas a cada estado-acción.
                                        #  Un diccionario que realiza un seguimiento de cuántas veces se ha visitado cada par estado-acción.
        self.total_visits = 0  # Contador total de visitas a estados.
                                #  Cuenta el número total de pasos que ha dado el agente.

        # Grafo para registrar estados visitados
        self.state_graph = defaultdict(set)  # Un diccionario que representa el grafo del entorno.
                                # Las claves son los estados y los valores son conjuntos de tuplas (next_state, action, reward) que indican a qué estados se puede llegar desde un estado dado.

    def select_action(self, state):
        """
        Selecciona una acción según la estrategia de exploración/explotación.

        Args:
            state: Estado actual.  Representa la situación actual del agente en el entorno.

        Returns:
            Acción seleccionada.  Un entero que indica la acción que el agente debe tomar.
        """
        if self.exploration_strategy == 'epsilon_greedy':
            return self._epsilon_greedy(state)
        elif self.exploration_strategy == 'ucb':
            return self._ucb(state)
        elif self.exploration_strategy == 'boltzmann':
            return self._boltzmann(state)
        else:
            raise ValueError("Estrategia no válida")  # Lanza un error si la estrategia no es válida.

    def _epsilon_greedy(self, state):
        """
        Implementación de la estrategia ε-greedy.

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        if random.random() < self.epsilon:  # Exploración: elige una acción aleatoria con probabilidad ε.
            return random.choice(range(self.env.action_space.n))  # Explora una acción aleatoria.
        return np.argmax(self.q_values[state])  # Explotación: elige la acción con el valor Q más alto.

    def _ucb(self, state):
        """
        Implementación de Upper Confidence Bound (UCB).

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        if self.visit_counts[state].sum() == 0:  # Si no se han visitado acciones desde este estado, explora.
            return random.choice(range(self.env.action_space.n))  # Explora una acción aleatoria.

        ucb_values = []
        for action in range(self.env.action_space.n):
            if self.visit_counts[state][action] == 0:  # Prioriza acciones no exploradas.
                return action  # Explora acciones que no han sido probadas.

            # Cálculo de UCB: combina la estimación del valor (explotación) con un término de incertidumbre (exploración).
            exploitation = self.q_values[state][action]  # El valor Q estimado para la acción.
            exploration = self.ucb_c * np.sqrt(np.log(self.total_visits) / self.visit_counts[state][action])  # Término de exploración que depende de πόσο se ha visitado la acción.
            ucb_values.append(exploitation + exploration)  # Suma la explotación y la exploración para obtener el valor UCB.

        return np.argmax(ucb_values)  # Selecciona la acción con el valor UCB más alto.

    def _boltzmann(self, state):
        """
        Implementación de la estrategia Boltzmann (Softmax).

        Args:
            state: Estado actual.

        Returns:
            Acción seleccionada.
        """
        q_values = self.q_values[state]
        probabilities = np.exp(q_values / self.temperature) / np.sum(np.exp(q_values / self.temperature))  # Calcula las probabilidades de cada acción usando softmax y la temperatura.
        return np.random.choice(range(self.env.action_space.n), p=probabilities)  # Muestrea una acción según las probabilidades.

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
        self.visit_counts[state][action] += 1  # Incrementa el contador de visitas para el par estado-acción.
        self.total_visits += 1  # Incrementa el contador total de visitas.

        # Actualizar valores Q usando una tasa de aprendizaje adaptativa
        alpha = 1 / self.visit_counts[state][action]  # Tasa de aprendizaje que disminuye con el número de visitas.
        self.q_values[state][action] += alpha * (reward + np.max(self.q_values[next_state]) - self.q_values[state][action])  # Actualiza el valor Q usando la diferencia temporal (TD).

        # Actualizar grafo de estados
        self.state_graph[state].add((next_state, action, reward))  # Agrega una arista al grafo de estados que representa la transición del estado actual al siguiente estado al tomar la acción.

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
        queue = deque()  # Inicializa una cola para la búsqueda en anchura.
        queue.append((start_state, [], 0))  # Agrega el estado inicial a la cola con un camino vacío y una recompensa de 0.  (estado, camino, recompensa acumulada)
        best_path = []  # Inicializa la variable para almacenar el mejor camino encontrado.
        max_reward = -np.inf  # Inicializa la variable para almacenar la máxima recompensa encontrada.

        while queue:
            current_state, path, current_reward = queue.popleft()  # Usa popleft() para FIFO.  Obtiene el siguiente estado de la cola.

            if current_state == goal_state:  # Si se llega al objetivo
                if current_reward > max_reward:  # Si la recompensa actual es mayor que la máxima recompensa encontrada hasta ahora
                    max_reward = current_reward  # Actualiza la máxima recompensa.
                    best_path = path  # Actualiza el mejor camino.
                continue  # Pasa a la siguiente iteración del bucle.

            for (next_state, action, reward) in self.state_graph.get(current_state, []):  # Itera sobre los posibles estados siguientes desde el estado actual.
                if next_state not in [s for s, _, _ in path]:  # Evitar ciclos.  Comprueba si el siguiente estado no está ya en el camino actual.
                    new_path = path + [(current_state, action)]  # Crea un nuevo camino agregando el estado actual y la acción tomada.
                    new_reward = current_reward + reward  # Calcula la nueva recompensa acumulada.
                    queue.append((next_state, new_path, new_reward))  # Agrega el nuevo estado a la cola.

        return best_path  # Devuelve el mejor camino encontrado.

# Clase del entorno del laberinto
class MazeEnv:
    """Entorno de laberinto para probar las estrategias."""
    def __init__(self, size=5):
        self.size = size  # Tamaño del laberinto (size x size)
        self.action_space = type('ActionSpace', (), {'n': 4})()  # 4 acciones posibles
        self.observation_space = type('ObservationSpace', (), {'n': size*size})()  # Número total de estados posibles
        self.goal = (size-1, size-1)  # Meta en la esquina inferior derecha
        self.reset()  # Inicializa el laberinto
    
    def reset(self):
        """Reinicia el entorno a la posición inicial."""
        self.state = (0, 0)  # El agente comienza en la esquina superior izquierda
        return self._get_state()  # Devuelve el estado inicial

    def _get_state(self):
        """Convierte coordenadas (x, y) a un estado único."""
        x, y = self.state  # Obtiene las coordenadas x e y del agente
        return x * self.size + y  # Convierte las coordenadas a un número entero único

    def step(self, action):
        """
        Ejecuta una acción y devuelve el resultado.

        Args:
            action: Acción a ejecutar.  0: arriba, 1: abajo, 2: izquierda, 3: derecha

        Returns:
            next_state: Estado siguiente.
            reward: Recompensa obtenida.
            done: Si se alcanzó la meta.
            info: Información adicional (vacío en este caso).
        """
        x, y = self.state  # Obtiene la posición actual del agente
        reward = -0.1  # Penalización por movimiento para fomentar caminos cortos
        done = False  # Inicialmente, el episodio no ha terminado

        # Movimientos según la acción
        if action == 0: x = max(x-1, 0)      # Arriba, no salir del borde superior
        elif action == 1: x = min(x+1, self.size-1)  # Abajo, no salir del borde inferior
        elif action == 2: y = max(y-1, 0)     # Izquierda, no salir del borde izquierdo
        elif action == 3: y = min(y+1, self.size-1)  # Derecha, no salir del borde derecho

        self.state = (x, y)  # Actualiza la posición del agente

        # Recompensa por llegar a la meta
        if (x, y) == self.goal:
            reward = 10  # Recompensa grande por alcanzar la meta
            done = True  # El episodio termina al llegar a la meta

        return self._get_state(), reward, done, {}  # Devuelve el siguiente estado, la recompensa, si el episodio terminó y información adicional.

# Función para comparar estrategias
def compare_strategies():
    """
    Compara diferentes estrategias de exploración/explotación en el entorno.
    """
    env = MazeEnv(size=5)  # Crear entorno de laberinto 5x5
    strategies = [
        ('epsilon_greedy (ε=0.1)', 'epsilon_greedy', {'epsilon': 0.1}),  # ε-greedy con ε = 0.1
        ('epsilon_greedy (ε=0.3)', 'epsilon_greedy', {'epsilon': 0.3}),  # ε-greedy con ε = 0.3
        ('UCB (c=2)', 'ucb', {'ucb_c': 2}),  # UCB con c = 2
        ('Boltzmann (τ=0.5)', 'boltzmann', {'temperature': 0.5}),  # Boltzmann con temperatura = 0.5
    ]

    results = {}  # Diccionario para almacenar los resultados de cada estrategia

    for name, strategy, params in strategies:  # Itera sobre cada estrategia
        # Crear agente con la estrategia actual
        agent = ExplorationExploitationAgent(env, exploration_strategy=strategy, **params)
        rewards = []  # Lista para almacenar las recompensas obtenidas en cada episodio de entrenamiento

        # Entrenamiento
        for episode in range(500):  # Entrena al agente durante 500 episodios
            state = env.reset()  # Reinicia el entorno al inicio de cada episodio
            done = False  # Inicializa la variable para indicar si el episodio ha terminado
            total_reward = 0  # Inicializa la recompensa total para el episodio actual

            while not done:
                action = agent.select_action(state)  # Selecciona una acción usando la estrategia actual
                next_state, reward, done, _ = env.step(action)  # Ejecuta la acción en el entorno
                agent.update(state, action, reward, next_state)  # Actualiza los valores Q y el grafo de estados
                state = next_state  # Actualiza el estado actual
                total_reward += reward  # Acumula la recompensa

            rewards.append(total_reward)  # Almacena la recompensa total del episodio

        # Evaluación
        test_rewards = []  # Lista para almacenar las recompensas obtenidas en cada episodio de prueba
        for _ in range(100):  # Evalúa al agente durante 100 episodios
            state = env.reset()  # Reinicia el entorno al inicio de cada episodio de prueba
            done = False  # Inicializa la variable para indicar si el episodio ha terminado
            total_reward = 0  # Inicializa la recompensa total para el episodio de prueba

            while not done:
                action = np.argmax(agent.q_values[state])  # Solo explotación: elige la mejor acción según los valores Q aprendidos
                next_state, reward, done, _ = env.step(action)  # Ejecuta la acción en el entorno
                state = next_state  # Actualiza el estado actual
                total_reward += reward  # Acumula la recompensa

            test_rewards.append(total_reward)  # Almacena la recompensa total del episodio de prueba

        # Guardar resultados
        results[name] = {  # Almacena los resultados para la estrategia actual
            'training': rewards,  # Recompensas obtenidas durante el entrenamiento
            'testing': np.mean(test_rewards)  # Recompensa promedio obtenida durante la prueba
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
    plt.figure(figsize=(12, 6))  # Crea una nueva figura para el gráfico
    for name, data in results.items():  # Itera sobre los resultados de cada estrategia
        plt.plot(np.convolve(data['training'], np.ones(10)/10, mode='valid'), label=name)  # Grafica la recompensa promedio suavizada (usando una ventana móvil de 10 episodios)
    plt.title("Comparación de estrategias de exploración/explotación")  # Establece el título del gráfico
    plt.xlabel("Episodio")  # Establece la etiqueta del eje x
    plt.ylabel("Recompensa promedio (suavizada)")  # Establece la etiqueta del eje y
    plt.legend()  # Muestra la leyenda del gráfico
    plt.grid()  # Muestra la cuadrícula en el gráfico
    plt.show()  # Muestra el gráfico

# Punto de entrada principal
if __name__ == "__main__":
    compare_strategies()  # Llama a la función para comparar las estrategias
