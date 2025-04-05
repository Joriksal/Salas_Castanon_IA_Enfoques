import numpy as np
from collections import defaultdict
import random
import time
import matplotlib.pyplot as plt

# Clase que representa al agente que busca la política óptima
class PolicySearchAgent:
    def __init__(self, env, gamma=0.99, learning_rate=0.1):
        self.env = env  # Entorno en el que opera el agente
        self.gamma = gamma  # Factor de descuento para las recompensas futuras
        self.alpha = learning_rate  # Tasa de aprendizaje
        # Política inicial: asigna acciones aleatorias a cada estado
        self.policy = defaultdict(lambda: random.choice(range(env.action_space.n)))
        # Valores de los estados (inicialmente 0)
        self.state_values = defaultdict(float)
        # Grafo de transiciones: almacena las transiciones entre estados
        self.transition_graph = defaultdict(lambda: defaultdict(list))
        # Contador de visitas a cada estado
        self.visit_counts = defaultdict(int)
    
    # Construye el grafo de transiciones explorando el entorno
    def build_transition_graph(self, num_episodes=10):
        print("Construyendo grafo de transiciones...")
        start_time = time.time()
        
        for ep in range(num_episodes):  # Ejecuta varios episodios
            state = self.env.reset()  # Reinicia el entorno
            done = False
            
            while not done:  # Mientras no se alcance un estado terminal
                action = self.policy[state]  # Selecciona una acción según la política actual
                next_state, reward, done, _ = self.env.step(action)  # Ejecuta la acción
                
                # Registra la transición en el grafo
                self.transition_graph[state][action].append((next_state, reward))
                self.visit_counts[state] += 1  # Incrementa el contador de visitas
                state = next_state  # Actualiza el estado actual
        
        print(f"Grafo construido en {time.time()-start_time:.2f}s. Estados descubiertos: {len(self.transition_graph)}")

    # Evalúa la política actual calculando los valores de los estados
    def policy_evaluation(self, max_iter=100, theta=1e-3):
        print("Evaluando política actual...")
        for i in range(max_iter):  # Itera hasta un máximo de iteraciones
            delta = 0  # Diferencia máxima entre valores antiguos y nuevos
            for state in self.transition_graph:  # Para cada estado en el grafo
                old_value = self.state_values[state]  # Valor actual del estado
                total = 0
                count = 0
                
                # Calcula el valor promedio de las recompensas futuras
                for action in self.transition_graph[state]:
                    for next_state, reward in self.transition_graph[state][action]:
                        total += reward + self.gamma * self.state_values[next_state]
                        count += 1
                
                if count > 0:
                    self.state_values[state] = total / count  # Actualiza el valor del estado
                    delta = max(delta, abs(old_value - self.state_values[state]))
            
            if delta < theta:  # Si los valores han convergido, detiene la evaluación
                print(f"Evaluación convergió en iteración {i+1}")
                break
    
    # Mejora la política actual seleccionando las mejores acciones
    def policy_improvement(self):
        print("Mejorando política...")
        policy_updated = False
        
        for state in self.transition_graph:  # Para cada estado en el grafo
            old_action = self.policy[state]  # Acción actual según la política
            action_values = []
            
            # Calcula el valor esperado para cada acción
            for action in range(self.env.action_space.n):
                if action not in self.transition_graph[state]:
                    action_values.append(-np.inf)  # Acción no válida
                    continue
                    
                total = 0
                for next_state, reward in self.transition_graph[state][action]:
                    total += reward + self.gamma * self.state_values[next_state]
                
                action_values.append(total / len(self.transition_graph[state][action]))
            
            # Selecciona la acción con el mayor valor esperado
            new_action = np.argmax(action_values)
            if new_action != old_action:  # Si la acción cambia, actualiza la política
                self.policy[state] = new_action
                policy_updated = True
        
        return policy_updated  # Devuelve si la política fue actualizada
    
    # Entrena al agente iterando entre exploración, evaluación y mejora
    def train(self, max_iter=10):
        print("\nIniciando entrenamiento...")
        for i in range(max_iter):
            print(f"\n--- Iteración {i+1}/{max_iter} ---")
            
            # Paso 1: Exploración
            self.build_transition_graph(num_episodes=5)
            
            # Paso 2: Evaluación
            self.policy_evaluation(max_iter=20)
            
            # Paso 3: Mejora
            if not self.policy_improvement():
                print("Política no mejoró - terminando entrenamiento")
                break
            
            # Mostrar progreso
            if (i+1) % 2 == 0:
                test_reward = self.test_policy()
                print(f"Recompensa promedio en prueba: {test_reward:.2f}")
    
    # Prueba la política actual y calcula la recompensa promedio
    def test_policy(self, num_tests=10):
        total_reward = 0
        for _ in range(num_tests):
            state = self.env.reset()
            done = False
            episode_reward = 0
            
            while not done:
                action = self.policy[state]
                state, reward, done, _ = self.env.step(action)
                episode_reward += reward
            
            total_reward += episode_reward
        
        return total_reward / num_tests
    
    # Encuentra el camino óptimo desde un estado inicial hasta uno objetivo
    def find_optimal_path(self, start_state, goal_state, max_steps=50):
        path = []
        current_state = start_state
        steps = 0
        
        while current_state != goal_state and steps < max_steps:
            path.append(current_state)
            action = self.policy[current_state]
            _, current_state, _, _ = self.env.step(action)
            steps += 1
        
        if current_state == goal_state:
            path.append(goal_state)
        
        return path

# Clase que representa el entorno GridWorld
class GridWorldEnv:
    def __init__(self, size=5):
        self.size = size  # Tamaño del grid
        self.action_space = type('ActionSpace', (), {'n': 4})()  # 4 acciones posibles (arriba, abajo, izquierda, derecha)
        self.observation_space = type('ObservationSpace', (), {'n': size*size})()  # Número total de estados
        self.goal = (size-1, size-1)  # Estado objetivo
        self.reset()
    
    def reset(self):
        self.state = (0, 0)  # Estado inicial
        return self._get_state()
    
    def _get_state(self):
        x, y = self.state
        return x * self.size + y  # Convierte coordenadas (x, y) a un índice único
    
    def step(self, action):
        x, y = self.state
        reward = -0.1  # Penalización por cada paso
        done = False
        
        # Actualiza la posición según la acción
        if action == 0: x = max(x-1, 0)  # Arriba
        elif action == 1: x = min(x+1, self.size-1)  # Abajo
        elif action == 2: y = max(y-1, 0)  # Izquierda
        elif action == 3: y = min(y+1, self.size-1)  # Derecha
        
        self.state = (x, y)
        state = self._get_state()
        
        # Si alcanza el objetivo, recompensa alta y termina el episodio
        if (x, y) == self.goal:
            reward = 10
            done = True
        
        return state, reward, done, {}

# Visualiza el camino óptimo en el grid
def visualize_path(env, path):
    grid = np.zeros((env.size, env.size))
    
    for state in path:
        x = state // env.size
        y = state % env.size
        grid[x, y] = 1  # Marca el camino
    
    grid[0, 0] = 2  # Estado inicial
    grid[env.size-1, env.size-1] = 3  # Estado objetivo
    
    plt.figure(figsize=(6, 6))
    plt.imshow(grid, cmap='viridis')
    plt.colorbar(ticks=[0, 1, 2, 3])
    plt.title("Camino óptimo encontrado")
    plt.show()

# Función principal
def main():
    # Configuración del entorno y agente
    env = GridWorldEnv(size=5)
    agent = PolicySearchAgent(env)
    
    # Entrenamiento del agente
    agent.train(max_iter=5)
    
    # Prueba final de la política
    test_reward = agent.test_policy(num_tests=20)
    print(f"\nRecompensa final promedio: {test_reward:.2f}")
    
    # Encuentra y muestra el camino óptimo
    start_state = env.reset()
    goal_state = env.size * env.size - 1
    path = agent.find_optimal_path(start_state, goal_state)
    
    print("\nCamino óptimo:")
    for state in path:
        x, y = state // env.size, state % env.size
        print(f"({x},{y})", end=" -> ")
    print("Meta")
    
    # Visualización del camino
    visualize_path(env, path)

# Ejecuta el programa principal
if __name__ == "__main__":
    main()