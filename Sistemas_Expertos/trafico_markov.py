# Importamos librerías necesarias
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)
plt.style.use('ggplot')

# =============================================
# 1. DEFINICIÓN DEL PROCESO DE MARKOV (TRÁFICO)
# =============================================

class MarkovProcess:
    def __init__(self, transition_matrix, states):
        self.P = np.array(transition_matrix)
        self.states = states
        self.state_index = {s: i for i, s in enumerate(states)}
        self._validate_matrix()
    
    def _validate_matrix(self):
        if not np.allclose(self.P.sum(axis=1), 1):
            raise ValueError("Las filas de la matriz deben sumar 1")
    
    def next_state(self, current_state):
        current_idx = self.state_index[current_state]
        next_idx = np.random.choice(len(self.states), p=self.P[current_idx])
        return self.states[next_idx]
    
    def simulate(self, initial_state, n_steps=10):
        trajectory = [initial_state]
        current_state = initial_state
        for _ in range(n_steps - 1):
            current_state = self.next_state(current_state)
            trajectory.append(current_state)
        return trajectory
    
    def stationary_distribution(self, max_iter=1000, tol=1e-6):
        pi = np.ones(len(self.states)) / len(self.states)
        for _ in range(max_iter):
            new_pi = pi @ self.P
            if np.linalg.norm(new_pi - pi) < tol:
                break
            pi = new_pi
        return {state: prob for state, prob in zip(self.states, pi)}

# =============================================
# 2. MODELO DE TRÁFICO
# =============================================

traffic_model = MarkovProcess(
    transition_matrix=[
        [0.6, 0.3, 0.1],  # Fluido: Probabilidades de Fluido, Moderado, Congestionado
        [0.3, 0.5, 0.2],  # Moderado
        [0.1, 0.4, 0.5]   # Congestionado
    ],
    states=['Fluido', 'Moderado', 'Congestionado']
)

# Simulación de tráfico durante 12 horas
traffic_traj = traffic_model.simulate('Fluido', n_steps=12)
print("\nSimulación de tráfico durante 12 horas:")
print(" -> ".join(traffic_traj))

# Distribución estacionaria
stationary = traffic_model.stationary_distribution()
print("\nDistribución estacionaria del tráfico:")
for state, prob in stationary.items():
    print(f"{state}: {prob:.4f}")

# =============================================
# 3. MODELO OCULTO DE MARKOV (HMM) PARA EXPERIENCIA DEL CONDUCTOR
# =============================================

class HiddenMarkovModel:
    def __init__(self, transition_matrix, emission_matrix, states, observations):
        self.markov_chain = MarkovProcess(transition_matrix, states)
        self.B = np.array(emission_matrix)
        self.observations = observations
        self.obs_index = {o: i for i, o in enumerate(observations)}
    
    def generate_sequence(self, initial_state, n_steps):
        hidden_states = self.markov_chain.simulate(initial_state, n_steps)
        observed = []
        for state in hidden_states:
            state_idx = self.markov_chain.state_index[state]
            obs_idx = np.random.choice(len(self.observations), p=self.B[state_idx])
            observed.append(self.observations[obs_idx])
        return hidden_states, observed

# Definimos el HMM para tráfico
hmm_traffic = HiddenMarkovModel(
    transition_matrix=[
        [0.6, 0.3, 0.1],
        [0.3, 0.5, 0.2],
        [0.1, 0.4, 0.5]
    ],
    emission_matrix=[
        [0.8, 0.15, 0.05],  # Fluido: Viaje rápido, Viaje lento, Atasco
        [0.3, 0.5, 0.2],    # Moderado
        [0.05, 0.35, 0.6]   # Congestionado
    ],
    states=['Fluido', 'Moderado', 'Congestionado'],
    observations=['Viaje rápido', 'Viaje lento', 'Atasco']
)

# Simulamos la experiencia del conductor durante 12 horas
hidden, observed = hmm_traffic.generate_sequence('Fluido', 12)
print("\nHMM - Experiencia del conductor:")
print("Tráfico real:  ", " -> ".join(hidden))
print("Experiencia:   ", " -> ".join(observed))

# =============================================
# 4. VISUALIZACIÓN
# =============================================

def plot_markov_chain(model):
    plt.figure(figsize=(10, 6))
    for i, state in enumerate(model.states):
        for j, next_state in enumerate(model.states):
            prob = model.P[i, j]
            if prob > 0:
                plt.plot([i, j], [0, 0], 'o-', linewidth=prob*10, markersize=20, alpha=0.7,
                         label=f"{state}→{next_state}: {prob:.2f}")
    plt.xticks(range(len(model.states)), model.states)
    plt.title("Diagrama de transición del tráfico")
    plt.grid(True, alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.show()

plot_markov_chain(traffic_model)
