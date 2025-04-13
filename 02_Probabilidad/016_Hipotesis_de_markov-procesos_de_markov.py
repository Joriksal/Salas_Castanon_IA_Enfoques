import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# Configuración
np.random.seed(42)
plt.style.use('ggplot')

# =============================================
# 1. DEFINICIÓN DE UN PROCESO DE MARKOV
# =============================================

class MarkovProcess:
    """Implementa un proceso de Markov discreto en tiempo finito"""
    
    def __init__(self, transition_matrix, states):
        """
        Args:
            transition_matrix: Matriz de transición (n x n)
            states: Lista de nombres/identificadores de estados
        """
        self.P = np.array(transition_matrix)
        self.states = states
        self.state_index = {s:i for i,s in enumerate(states)}
        
        # Validar matriz de transición
        self._validate_matrix()
    
    def _validate_matrix(self):
        """Verifica que la matriz sea estocástica"""
        if not np.allclose(self.P.sum(axis=1), 1):
            raise ValueError("Las filas de la matriz deben sumar 1")
    
    def next_state(self, current_state):
        """Transición al siguiente estado"""
        current_idx = self.state_index[current_state]
        next_idx = np.random.choice(len(self.states), p=self.P[current_idx])
        return self.states[next_idx]
    
    def simulate(self, initial_state, n_steps=10):
        """Simula una trayectoria del proceso"""
        trajectory = [initial_state]
        current_state = initial_state
        
        for _ in range(n_steps-1):
            current_state = self.next_state(current_state)
            trajectory.append(current_state)
            
        return trajectory
    
    def stationary_distribution(self, max_iter=1000, tol=1e-6):
        """Calcula la distribución estacionaria mediante iteración"""
        pi = np.ones(len(self.states)) / len(self.states)  # Distribución inicial uniforme
        
        for _ in range(max_iter):
            new_pi = pi @ self.P
            if np.linalg.norm(new_pi - pi) < tol:
                break
            pi = new_pi
        
        return {state: prob for state, prob in zip(self.states, pi)}

# =============================================
# 2. EJEMPLO: MODELO DEL CLIMA
# =============================================

# Definimos el proceso de Markov para el clima
weather_model = MarkovProcess(
    transition_matrix=[
        [0.7, 0.2, 0.1],  # Soleado
        [0.3, 0.4, 0.3],  # Nublado
        [0.2, 0.3, 0.5]   # Lluvioso
    ],
    states=['Soleado', 'Nublado', 'Lluvioso']
)

# Simulación
weather_traj = weather_model.simulate('Soleado', n_steps=20)

# Resultados
print("\nSimulación del clima para 20 días:")
print(" -> ".join(weather_traj))

# Distribución estacionaria
stationary = weather_model.stationary_distribution()
print("\nDistribución estacionaria:")
for state, prob in stationary.items():
    print(f"{state}: {prob:.4f}")

# =============================================
# 3. VISUALIZACIÓN DE LA CADENA DE MARKOV
# =============================================

def plot_markov_chain(model):
    """Visualiza la cadena de Markov con gráfico de transiciones"""
    plt.figure(figsize=(10, 6))
    
    # Crear gráfico dirigido (simplificado)
    for i, state in enumerate(model.states):
        for j, next_state in enumerate(model.states):
            prob = model.P[i,j]
            if prob > 0:
                plt.plot([i, j], [0, 0], 'o-', 
                        linewidth=prob*10, 
                        markersize=20,
                        alpha=0.7,
                        label=f"{state}→{next_state}: {prob:.2f}")
    
    plt.xticks(range(len(model.states)), model.states)
    plt.title("Diagrama de Transiciones de la Cadena de Markov")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_markov_chain(weather_model)

# =============================================
# 4. APLICACIÓN EN IA: MODELO DE COMPORTAMIENTO DE USUARIO
# =============================================

# Modelo de interacción de usuario en una app
user_behavior = MarkovProcess(
    transition_matrix=[
        [0.6, 0.3, 0.1, 0.0],  # Inicio
        [0.2, 0.5, 0.2, 0.1],  # Exploración
        [0.1, 0.1, 0.7, 0.1],  # Compra
        [0.0, 0.0, 0.0, 1.0]   # Salida (estado absorbente)
    ],
    states=['Inicio', 'Exploración', 'Compra', 'Salida']
)

# Simulación de trayectorias de usuario
print("\nSimulaciones de comportamiento de usuario:")
for _ in range(5):
    traj = user_behavior.simulate('Inicio', 10)
    print(" -> ".join(traj))

# Probabilidad de absorción (llegar a 'Salida')
def absorption_probability(model, target_state, n_simulations=1000):
    """Estima probabilidad de alcanzar un estado absorbente"""
    count = 0
    for _ in range(n_simulations):
        traj = model.simulate('Inicio', 50)  # Número máximo de pasos
        if target_state in traj:
            count += 1
    return count / n_simulations

prob_exit = absorption_probability(user_behavior, 'Salida')
print(f"\nProbabilidad de que un usuario eventualmente salga: {prob_exit:.2%}")

# =============================================
# 5. EXTENSIÓN: PROCESOS DE MARKOV OCULTOS (HMM)
# =============================================

class HiddenMarkovModel:
    """Implementación básica de un Modelo Oculto de Markov (HMM)"""
    
    def __init__(self, transition_matrix, emission_matrix, states, observations):
        """
        Inicializa el HMM con una cadena de Markov y una matriz de emisión.
        
        Args:
            transition_matrix: Matriz de transición entre estados ocultos.
            emission_matrix: Matriz de emisión que relaciona estados ocultos con observaciones.
            states: Lista de estados ocultos.
            observations: Lista de posibles observaciones.
        """
        self.markov_chain = MarkovProcess(transition_matrix, states)  # Cadena de Markov subyacente
        self.B = np.array(emission_matrix)  # Matriz de emisión
        self.observations = observations  # Lista de observaciones posibles
        self.obs_index = {o: i for i, o in enumerate(observations)}  # Índices de observaciones
        
    def generate_sequence(self, initial_state, n_steps):
        """
        Genera una secuencia de estados ocultos y observaciones.
        
        Args:
            initial_state: Estado inicial de la cadena de Markov.
            n_steps: Número de pasos a simular.
        
        Returns:
            hidden_states: Lista de estados ocultos generados.
            observed: Lista de observaciones generadas.
        """
        # Generar la secuencia de estados ocultos usando la cadena de Markov
        hidden_states = self.markov_chain.simulate(initial_state, n_steps)
        observed = []  # Lista para almacenar las observaciones generadas
        
        # Para cada estado oculto, generar una observación basada en la matriz de emisión
        for state in hidden_states:
            state_idx = self.markov_chain.state_index[state]  # Índice del estado oculto actual
            # Seleccionar una observación con base en las probabilidades de emisión
            obs_idx = np.random.choice(len(self.observations), p=self.B[state_idx])
            observed.append(self.observations[obs_idx])  # Agregar la observación generada
            
        return hidden_states, observed  # Retornar estados ocultos y observaciones

# Ejemplo HMM: Clima con observaciones (actividades)
hmm_weather = HiddenMarkovModel(
    transition_matrix=[
        [0.7, 0.2, 0.1],  # Soleado: Probabilidades de transición a Soleado, Nublado, Lluvioso
        [0.3, 0.4, 0.3],  # Nublado: Probabilidades de transición a Soleado, Nublado, Lluvioso
        [0.2, 0.3, 0.5]   # Lluvioso: Probabilidades de transición a Soleado, Nublado, Lluvioso
    ],
    emission_matrix=[
        [0.8, 0.1, 0.1],  # Soleado: Probabilidades de Paseo, Leer, Cine
        [0.3, 0.4, 0.3],  # Nublado: Probabilidades de Paseo, Leer, Cine
        [0.1, 0.2, 0.7]   # Lluvioso: Probabilidades de Paseo, Leer, Cine
    ],
    states=['Soleado', 'Nublado', 'Lluvioso'],  # Estados ocultos
    observations=['Paseo', 'Leer', 'Cine']     # Observaciones posibles
)

# Generar una secuencia observada a partir de un estado inicial
hidden, observed = hmm_weather.generate_sequence('Soleado', 10)

# Imprimir los resultados
print("\nModelo de Markov Oculto:")
print("Estados reales:", " -> ".join(hidden))  # Secuencia de estados ocultos generados
print("Observaciones:", " -> ".join(observed))  # Secuencia de observaciones generadas