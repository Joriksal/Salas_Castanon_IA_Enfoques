# Importamos las librerías necesarias
import numpy as np  # Librería para cálculos numéricos y manejo de matrices
import matplotlib.pyplot as plt  # Librería para generar gráficos y visualizaciones

# Configuración inicial
np.random.seed(42)  # Fijamos una semilla para reproducibilidad en los números aleatorios
plt.style.use('ggplot')  # Estilo de gráficos para que sean más visualmente atractivos

# =============================================
# 1. DEFINICIÓN DE UN PROCESO DE MARKOV
# =============================================

class MarkovProcess:
    """
    Implementa un proceso de Markov discreto en tiempo finito.
    Un proceso de Markov es un modelo matemático que describe un sistema que transita entre estados
    con probabilidades definidas, dependiendo únicamente del estado actual (propiedad de Markov).
    """
    
    def __init__(self, transition_matrix, states):
        """
        Constructor de la clase MarkovProcess.
        
        Args:
            transition_matrix: Matriz de transición (n x n), donde cada fila representa las probabilidades
                               de transición desde un estado a los demás.
            states: Lista de nombres o identificadores de los estados del sistema.
        """
        self.P = np.array(transition_matrix)  # Convertimos la matriz de transición a un arreglo de NumPy
        self.states = states  # Guardamos los nombres de los estados
        self.state_index = {s: i for i, s in enumerate(states)}  # Diccionario para mapear estados a índices
        
        # Validamos que la matriz de transición sea estocástica (las filas deben sumar 1)
        self._validate_matrix()
    
    def _validate_matrix(self):
        """
        Verifica que la matriz de transición sea válida.
        Una matriz de transición es válida si cada fila suma exactamente 1.
        """
        if not np.allclose(self.P.sum(axis=1), 1):  # np.allclose verifica igualdad con tolerancia numérica
            raise ValueError("Las filas de la matriz deben sumar 1")  # Lanza un error si no es válida
    
    def next_state(self, current_state):
        """
        Calcula el siguiente estado del sistema basado en el estado actual.
        
        Args:
            current_state: El estado actual del sistema.
        
        Returns:
            El siguiente estado del sistema.
        """
        current_idx = self.state_index[current_state]  # Obtenemos el índice del estado actual
        # Elegimos el siguiente estado basado en las probabilidades de la fila correspondiente
        next_idx = np.random.choice(len(self.states), p=self.P[current_idx])
        return self.states[next_idx]  # Retornamos el nombre del siguiente estado
    
    def simulate(self, initial_state, n_steps=10):
        """
        Simula una trayectoria del proceso de Markov.
        
        Args:
            initial_state: Estado inicial del sistema.
            n_steps: Número de pasos a simular.
        
        Returns:
            Una lista con la trayectoria de estados visitados.
        """
        trajectory = [initial_state]  # Iniciamos la trayectoria con el estado inicial
        current_state = initial_state  # Establecemos el estado actual
        
        for _ in range(n_steps - 1):  # Iteramos n_steps - 1 veces (ya tenemos el estado inicial)
            current_state = self.next_state(current_state)  # Calculamos el siguiente estado
            trajectory.append(current_state)  # Lo añadimos a la trayectoria
            
        return trajectory  # Retornamos la lista de estados visitados
    
    def stationary_distribution(self, max_iter=1000, tol=1e-6):
        """
        Calcula la distribución estacionaria del proceso de Markov.
        La distribución estacionaria es un vector que describe las probabilidades de estar en cada estado
        cuando el sistema alcanza el equilibrio.
        
        Args:
            max_iter: Número máximo de iteraciones para aproximar la distribución.
            tol: Tolerancia para determinar la convergencia.
        
        Returns:
            Un diccionario con los estados y sus probabilidades estacionarias.
        """
        pi = np.ones(len(self.states)) / len(self.states)  # Inicializamos con una distribución uniforme
        
        for _ in range(max_iter):  # Iteramos hasta el número máximo de iteraciones
            new_pi = pi @ self.P  # Multiplicamos el vector actual por la matriz de transición
            if np.linalg.norm(new_pi - pi) < tol:  # Verificamos si la diferencia es menor que la tolerancia
                break  # Si converge, salimos del bucle
            pi = new_pi  # Actualizamos el vector de probabilidades
        
        # Retornamos un diccionario con los estados y sus probabilidades estacionarias
        return {state: prob for state, prob in zip(self.states, pi)}

# =============================================
# 2. EJEMPLO: MODELO DEL CLIMA
# =============================================

# Definimos un proceso de Markov para modelar el clima
weather_model = MarkovProcess(
    transition_matrix=[
        [0.7, 0.2, 0.1],  # Soleado: Probabilidades de transición a Soleado, Nublado, Lluvioso
        [0.3, 0.4, 0.3],  # Nublado: Probabilidades de transición a Soleado, Nublado, Lluvioso
        [0.2, 0.3, 0.5]   # Lluvioso: Probabilidades de transición a Soleado, Nublado, Lluvioso
    ],
    states=['Soleado', 'Nublado', 'Lluvioso']  # Estados posibles del clima
)

# Simulamos una trayectoria del clima comenzando en "Soleado" durante 20 días
weather_traj = weather_model.simulate('Soleado', n_steps=20)

# Mostramos los resultados de la simulación
print("\nSimulación del clima para 20 días:")
print(" -> ".join(weather_traj))  # Imprimimos la trayectoria como una cadena de estados

# Calculamos la distribución estacionaria del modelo del clima
stationary = weather_model.stationary_distribution()
print("\nDistribución estacionaria:")
for state, prob in stationary.items():
    print(f"{state}: {prob:.4f}")  # Mostramos las probabilidades con 4 decimales

# =============================================
# 3. VISUALIZACIÓN DE LA CADENA DE MARKOV
# =============================================

def plot_markov_chain(model):
    """
    Visualiza la cadena de Markov mediante un gráfico de transiciones.
    
    Args:
        model: Instancia de la clase MarkovProcess.
    """
    plt.figure(figsize=(10, 6))  # Configuramos el tamaño del gráfico
    
    # Dibujamos las transiciones entre estados
    for i, state in enumerate(model.states):
        for j, next_state in enumerate(model.states):
            prob = model.P[i, j]  # Probabilidad de transición de state a next_state
            if prob > 0:  # Solo graficamos transiciones con probabilidad mayor a 0
                plt.plot([i, j], [0, 0], 'o-', 
                         linewidth=prob * 10,  # Grosor proporcional a la probabilidad
                         markersize=20,
                         alpha=0.7,
                         label=f"{state}→{next_state}: {prob:.2f}")
    
    plt.xticks(range(len(model.states)), model.states)  # Etiquetas en el eje x
    plt.title("Diagrama de Transiciones de la Cadena de Markov")  # Título del gráfico
    plt.grid(True, alpha=0.3)  # Activamos la cuadrícula con transparencia
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')  # Leyenda fuera del gráfico
    plt.tight_layout()  # Ajustamos el diseño para evitar solapamientos
    plt.show()  # Mostramos el gráfico

# Llamamos a la función para visualizar el modelo del clima
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