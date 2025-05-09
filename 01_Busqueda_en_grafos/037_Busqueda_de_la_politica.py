import numpy as np  # Importa la librería NumPy para operaciones numéricas eficientes.
from collections import defaultdict  # Importa defaultdict para crear diccionarios con valores por defecto.
import random  # Importa la librería random para generar números aleatorios.
import time  # Importa la librería time para medir el tiempo de ejecución.
import matplotlib.pyplot as plt  # Importa Matplotlib para visualización de datos.

# Clase que representa al agente que busca la política óptima
class PolicySearchAgent:
    def __init__(self, env, gamma=0.99, learning_rate=0.1):
        """
        Inicializa el agente de búsqueda de políticas.

        Args:
            env: El entorno en el que opera el agente (p. ej., un laberinto).  Define el espacio de estados, las acciones y las reglas de transición.
            gamma: Factor de descuento para recompensas futuras.  Determina cuánto valora el agente las recompensas que recibirá en el futuro.
            learning_rate: Tasa de aprendizaje para actualizar los valores de los estados.  Controla cuánto se ajustan las estimaciones de valor en función de nueva información.
        """
        self.env = env  # Guarda una referencia al entorno.
        self.gamma = gamma  # Factor de descuento (importancia de recompensas futuras).
        self.alpha = learning_rate  # Tasa de aprendizaje (cuánto ajustar los valores).

        # Política inicial: asigna acciones aleatorias a cada estado.
        # Un diccionario donde las claves son los estados y los valores son las acciones.
        self.policy = defaultdict(lambda: random.choice(range(env.action_space.n)))
        # Por defecto, a cada estado se le asigna una acción aleatoria al inicio.  A medida que el agente aprende, esta política se actualiza.

        # Valores de los estados (inicialmente 0).
        # Un diccionario donde las claves son los estados y los valores son las estimaciones de su valor.
        self.state_values = defaultdict(float)
        # Almacena la estimación de cuánta recompensa a largo plazo puede esperar el agente estando en un estado dado. Inicialmente, estos valores se establecen en 0.

        # Grafo de transiciones: almacena las transiciones entre estados.
        # Un diccionario de diccionarios: el primer nivel mapea estados a acciones,
        # y el segundo nivel mapea acciones a listas de tuplas (next_state, reward).
        self.transition_graph = defaultdict(lambda: defaultdict(list))
        # Este grafo representa el conocimiento del agente sobre el entorno.  Mantiene un registro de qué estados se alcanzan al tomar una acción en un estado dado y qué recompensa se recibe.

        # Contador de visitas a cada estado
        self.visit_counts = defaultdict(int)
        # Realiza un seguimiento de cuántas veces el agente ha visitado cada estado. Esto puede ser útil para análisis o para modificar el comportamiento del agente.

    # Construye el grafo de transiciones explorando el entorno
    def build_transition_graph(self, num_episodes=10):
        """
        Construye un grafo de transiciones explorando el entorno durante varios episodios.

        Args:
            num_episodes: Número de episodios a ejecutar para construir el grafo.  Controla cuánto explora el agente el entorno.
        """
        print("Construyendo grafo de transiciones...")
        start_time = time.time()  # Registra el tiempo de inicio.

        for ep in range(num_episodes):  # Ejecuta varios episodios.
            state = self.env.reset()  # Reinicia el entorno al inicio de cada episodio.
            done = False  # Indica si el episodio ha terminado.

            while not done:  # Mientras el episodio no termine.
                action = self.policy[state]  # Selecciona una acción según la política actual.
                next_state, reward, done, _ = self.env.step(action)  # Ejecuta la acción en el entorno.

                # Registra la transición en el grafo.
                self.transition_graph[state][action].append((next_state, reward))
                # Para el estado actual, la acción tomada lleva al agente al siguiente estado y le da una recompensa.
                self.visit_counts[state] += 1  # Incrementa el contador de visitas para el estado actual.
                state = next_state  # Actualiza el estado actual.

        print(f"Grafo construido en {time.time()-start_time:.2f}s. Estados descubiertos: {len(self.transition_graph)}")
        # Imprime cuánto tiempo tardó en construirse el grafo y cuántos estados únicos se descubrieron.

    # Evalúa la política actual calculando los valores de los estados
    def policy_evaluation(self, max_iter=100, theta=1e-3):
        """
        Evalúa la política actual iterativamente hasta que los valores de los estados convergen.

        Args:
            max_iter: Máximo número de iteraciones.  Evita que el algoritmo se ejecute indefinidamente si no converge.
            theta: Umbral para determinar la convergencia.  Define qué tan pequeño debe ser el cambio en los valores del estado para considerar que la evaluación ha terminado.
        """
        print("Evaluando política actual...")
        for i in range(max_iter):  # Itera hasta un máximo de iteraciones.
            delta = 0  # Mayor cambio en los valores de los estados en esta iteración.
            for state in self.transition_graph:  # Para cada estado en el grafo.
                old_value = self.state_values[state]  # Guarda el valor anterior del estado.
                total = 0
                count = 0

                # Calcula el valor esperado del estado actual usando la política actual.
                # Itera sobre todas las acciones posibles desde el estado actual y los posibles estados siguientes y recompensas para cada acción.
                for action in self.transition_graph[state]:
                    for next_state, reward in self.transition_graph[state][action]:
                        total += reward + self.gamma * self.state_values[next_state]
                        count += 1

                if count > 0:
                    self.state_values[state] = total / count  # Actualiza el valor del estado.
                    # El nuevo valor es el promedio ponderado de las recompensas inmediatas y el valor descontado de los estados siguientes.
                    delta = max(delta, abs(old_value - self.state_values[state]))  # Calcula el cambio absoluto.
                    # Lleva un registro del mayor cambio de valor de estado visto en esta iteración.

            if delta < theta:  # Si los cambios son pequeños, los valores han convergido.
                print(f"Evaluación convergió en iteración {i+1}")
                break  # Termina el bucle de evaluación.

    # Mejora la política actual seleccionando las mejores acciones
    def policy_improvement(self):
        """
        Mejora la política actual iterando sobre todos los estados y seleccionando la acción que maximiza el valor esperado.

        Returns:
            True si la política fue actualizada, False en caso contrario.
        """
        print("Mejorando política...")
        policy_updated = False  # Indica si la política fue actualizada en esta iteración.

        for state in self.transition_graph:  # Para cada estado en el grafo.
            old_action = self.policy[state]  # Guarda la acción anterior para este estado.
            action_values = []  # Lista para almacenar los valores esperados de cada acción.

            # Calcula el valor esperado de cada acción posible desde el estado actual.
            for action in range(self.env.action_space.n):
                if action not in self.transition_graph[state]:
                    action_values.append(-np.inf)  # Asigna un valor de -infinito a las acciones no válidas.
                    continue

                total = 0
                for next_state, reward in self.transition_graph[state][action]:
                    total += reward + self.gamma * self.state_values[next_state]

                action_values.append(total / len(self.transition_graph[state][action]))  # Calcula el valor promedio.
                # Para cada acción, calcula el valor esperado sumando las recompensas inmediatas y los valores descontados de los estados siguientes, y luego promediando.

            # Selecciona la acción con el mayor valor esperado.
            new_action = np.argmax(action_values)
            if new_action != old_action:  # Si la acción óptima es diferente de la acción anterior.
                self.policy[state] = new_action  # Actualiza la política para este estado.
                policy_updated = True  # Marca la política como actualizada.

        return policy_updated  # Devuelve si la política fue actualizada.

    # Entrena al agente iterando entre exploración, evaluación y mejora
    def train(self, max_iter=10):
        """
        Entrena al agente iterando entre exploración (construcción del grafo), evaluación de la política y mejora de la política.

        Args:
            max_iter: Número máximo de iteraciones de entrenamiento.  Controla cuántas veces se repite el ciclo de aprendizaje.
        """
        print("\nIniciando entrenamiento...")
        for i in range(max_iter):
            print(f"\n--- Iteración {i+1}/{max_iter} ---")

            # Paso 1: Exploración
            self.build_transition_graph(num_episodes=5)  # Construye/actualiza el grafo de transiciones.
            # El agente explora el entorno para aprender sobre su estructura y las consecuencias de sus acciones.

            # Paso 2: Evaluación
            self.policy_evaluation(max_iter=20)  # Evalúa la política actual.
            # Dado el conocimiento del agente sobre el entorno (el grafo de transiciones), determina qué tan buena es la política actual.

            # Paso 3: Mejora
            if not self.policy_improvement():  # Mejora la política actual.
                print("Política no mejoró - terminando entrenamiento")
                break  # Si la política no cambia, termina el entrenamiento.
            # El agente actualiza su política para seleccionar acciones que conduzcan a las recompensas a largo plazo más altas.

            # Mostrar progreso
            if (i+1) % 2 == 0:
                test_reward = self.test_policy()  # Prueba la política actual.
                print(f"Recompensa promedio en prueba: {test_reward:.2f}")
                # Cada dos iteraciones, el agente prueba qué tan bien está funcionando su política aprendida en el entorno.
                # Esto da una idea del progreso del agente a lo largo del tiempo.

    # Prueba la política actual y calcula la recompensa promedio
    def test_policy(self, num_tests=10):
        """
        Prueba la política aprendida ejecutando varios episodios de prueba y calcula la recompensa promedio.

        Args:
            num_tests: Número de episodios de prueba.  Controla la precisión de la estimación de la recompensa.
        
        Returns:
            Recompensa promedio obtenida durante las pruebas.
        """
        total_reward = 0
        for _ in range(num_tests):  # Ejecuta varios episodios de prueba.
            state = self.env.reset()  # Reinicia el entorno al inicio de cada episodio.
            done = False  # Indica si el episodio ha terminado.
            episode_reward = 0  # Recompensa total para el episodio actual.

            while not done:  # Mientras el episodio no termine.
                action = self.policy[state]  # Selecciona la acción según la política aprendida.
                state, reward, done, _ = self.env.step(action)  # Ejecuta la acción.
                episode_reward += reward  # Acumula la recompensa.

            total_reward += episode_reward  # Acumula la recompensa del episodio.

        return total_reward / num_tests  # Devuelve la recompensa promedio.

    # Encuentra el camino óptimo desde un estado inicial hasta uno objetivo
    def find_optimal_path(self, start_state, goal_state, max_steps=50):
        """
        Encuentra el camino óptimo desde un estado inicial hasta un estado objetivo siguiendo la política aprendida.

        Args:
            start_state: Estado inicial.
            goal_state: Estado objetivo.
            max_steps: Máximo número de pasos permitidos.  Evita que el agente se quede atascado en un bucle infinito.

        Returns:
            Una lista de estados que representan el camino óptimo, o una lista vacía si no se encuentra un camino.
        """
        path = []  # Lista para almacenar el camino.
        current_state = start_state  # Comienza desde el estado inicial.
        steps = 0  # Contador de pasos.

        while current_state != goal_state and steps < max_steps:  # Mientras no se alcance el objetivo y no se exceda el límite de pasos.
            path.append(current_state)  # Agrega el estado actual al camino.
            action = self.policy[current_state]  # Obtiene la acción recomendada por la política.
            _, current_state, _, _ = self.env.step(action)  # Ejecuta la acción y obtiene el siguiente estado.
            steps += 1  # Incrementa el contador de pasos.

        if current_state == goal_state:  # Si se alcanza el estado objetivo.
            path.append(goal_state)  # Agrega el estado objetivo al camino.

        return path  # Devuelve el camino encontrado.

# Clase que representa el entorno GridWorld
class GridWorldEnv:
    def __init__(self, size=5):
        """
        Inicializa el entorno GridWorld.

        Args:
            size: Tamaño del grid (size x size).  Determina el tamaño del laberinto.
        """
        self.size = size  # Tamaño del grid.
        self.action_space = type('ActionSpace', (), {'n': 4})()  # 4 acciones posibles (arriba, abajo, izquierda, derecha).
        # Define el espacio de acciones como un objeto con un atributo 'n' que especifica el número de acciones.
        self.observation_space = type('ObservationSpace', (), {'n': size*size})()  # Número total de estados.
        # Define el espacio de observaciones como un objeto con un atributo 'n' que especifica el número de estados posibles.
        self.goal = (size-1, size-1)  # Estado objetivo (esquina inferior derecha).
        self.reset()  # Inicializa el entorno.

    def reset(self):
        """Reinicia el entorno a la posición inicial."""
        self.state = (0, 0)  # El agente comienza en la esquina superior izquierda.
        return self._get_state()  # Devuelve el estado inicial.

    def _get_state(self):
        """Convierte coordenadas (x, y) a un índice de estado único."""
        x, y = self.state  # Obtiene las coordenadas x e y del agente.
        return x * self.size + y  # Convierte las coordenadas a un número entero único.
        # Esto es necesario porque los algoritmos de RL a menudo tratan los estados como números discretos.

    def step(self, action):
        """
        Ejecuta una acción en el entorno y devuelve el siguiente estado, la recompensa, si el episodio ha terminado y otra información.

        Args:
            action: La acción a ejecutar (0: arriba, 1: abajo, 2: izquierda, 3: derecha).

        Returns:
            Una tupla: (next_state, reward, done, info).
            next_state: El siguiente estado después de ejecutar la acción.
            reward: La recompensa obtenida por ejecutar la acción.
            done: Un booleano que indica si el episodio ha terminado.
            info: Un diccionario con información adicional (generalmente vacío).
        """
        x, y = self.state  # Obtiene la posición actual del agente.
        reward = -0.1  # Penalización por cada paso para fomentar caminos cortos.
        done = False  # Inicialmente, el episodio no ha terminado.

        # Actualiza la posición del agente según la acción tomada.
        if action == 0: x = max(x-1, 0)  # Arriba, no salir del borde superior.
        elif action == 1: x = min(x+1, self.size-1)  # Abajo, no salir del borde inferior.
        elif action == 2: y = max(y-1, 0)  # Izquierda, no salir del borde izquierdo.
        elif action == 3: y = min(y+1, self.size-1)  # Derecha, no salir del borde derecho.

        self.state = (x, y)  # Actualiza la posición del agente.
        state = self._get_state()  # Obtiene el índice del nuevo estado.

        # Si el agente alcanza el estado objetivo, el episodio termina y recibe una recompensa grande.
        if (x, y) == self.goal:
            reward = 10  # Recompensa por alcanzar el objetivo.
            done = True  # El episodio ha terminado.

        return state, reward, done, {}  # Devuelve el resultado de la acción.

# Visualiza el camino óptimo en el grid
def visualize_path(env, path):
    """
    Visualiza el camino óptimo en el entorno GridWorld usando matplotlib.

    Args:
        env: El entorno GridWorld.
        path: Lista de estados que representan el camino óptimo.
    """
    grid = np.zeros((env.size, env.size))  # Crea una matriz para representar el grid.
    # Inicializa una matriz 2D con ceros.  Esta matriz representará visualmente el laberinto.

    for state in path:  # Marca el camino en el grid.
        x = state // env.size  # Convierte el índice del estado a coordenadas x, y.
        y = state % env.size
        grid[x, y] = 1  # Marca la celda como parte del camino.
        # Itera a través de la lista de estados que forman el camino óptimo y marca las celdas correspondientes en la matriz del grid con el valor 1.

    grid[0, 0] = 2  # Marca el estado inicial con un valor diferente.
    grid[env.size-1, env.size-1] = 3  # Marca el estado objetivo con un valor diferente.
    # Marca el estado inicial con el valor 2 y el estado objetivo con el valor 3 para distinguirlos visualmente en la visualización.

    # Visualiza el grid con el camino marcado.
    plt.figure(figsize=(6, 6))  # Crea una figura de tamaño 6x6 pulgadas.
    plt.imshow(grid, cmap='viridis')  # Muestra la matriz como una imagen, usando un colormap 'viridis'.
    # Usa la función imshow de Matplotlib para mostrar la matriz del grid como una imagen.
    # El argumento cmap especifica el mapa de colores que se utilizará para la visualización.
    plt.colorbar(ticks=[0, 1, 2, 3])  # Añade una barra de color para interpretar los valores del grid.
    # Añade una barra de color a la visualización para mostrar qué valor de la matriz corresponde a qué color en la imagen.
    # El argumento ticks especifica las ubicaciones de las etiquetas en la barra de color.
    plt.title("Camino óptimo encontrado")  # Establece el título del gráfico.
    plt.show()  # Muestra el gráfico.

# Función principal
def main():
    """
    Función principal que crea el entorno, inicializa al agente, lo entrena y visualiza el camino óptimo.
    """
    # Configuración del entorno y agente
    env = GridWorldEnv(size=5)  # Crea un entorno GridWorld de 5x5.
    agent = PolicySearchAgent(env)  # Crea un agente de búsqueda de políticas.

    # Entrenamiento del agente
    agent.train(max_iter=5)  # Entrena al agente durante 5 iteraciones.
    # Llama al método de entrenamiento del agente para aprender una política para el entorno GridWorld.

    # Prueba final de la política
    test_reward = agent.test_policy(num_tests=20)  # Prueba la política aprendida en 20 episodios.
    print(f"\nRecompensa final promedio: {test_reward:.2f}")  # Imprime la recompensa promedio obtenida.
    # Llama al método de prueba del agente para evaluar el rendimiento de la política aprendida.

    # Encuentra y muestra el camino óptimo
    start_state = env.reset()  # Obtiene el estado inicial del entorno.
    goal_state = env.size * env.size - 1  # Calcula el índice del estado objetivo.
    path = agent.find_optimal_path(start_state, goal_state)  # Encuentra el camino óptimo usando la política del agente.
    # Llama al método del agente para encontrar la secuencia de estados que conduce desde el estado inicial al estado objetivo.

    print("\nCamino óptimo:")  # Imprime el camino óptimo encontrado.
    for state in path:
        x, y = state // env.size, state % env.size  # Convierte el índice del estado a coordenadas x, y.
        print(f"({x},{y})", end=" -> ")  # Imprime las coordenadas del estado.
    print("Meta")  # Indica que se ha alcanzado la meta.

    # Visualización del camino
    visualize_path(env, path)  # Visualiza el camino óptimo en el grid.
    # Llama a la función visualize_path para mostrar el camino óptimo en el entorno GridWorld.

# Ejecuta el programa principal
if __name__ == "__main__":
    main()  # Llama a la función principal cuando se ejecuta el script.
