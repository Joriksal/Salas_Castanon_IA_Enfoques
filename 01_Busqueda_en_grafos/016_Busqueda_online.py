import numpy as np  # Biblioteca para cálculos numéricos, usada para manejar valores Q.
from collections import defaultdict  # Estructura de datos para inicializar valores por defecto.
import random  # Biblioteca para generar números aleatorios, usada en la exploración.

class OnlineSearchAgent:
    def __init__(self, action_space, learning_rate=0.1, discount_factor=0.9, exploration_rate=0.2):
        """
        Inicializa un agente de búsqueda online basado en Q-learning.
        
        Args:
            action_space: Lista de acciones posibles (por ejemplo, ['arriba', 'abajo', 'izquierda', 'derecha']).
            learning_rate: Tasa de aprendizaje (0-1), controla cuánto se actualizan los valores Q.
            discount_factor: Factor de descuento (0-1), controla la importancia de recompensas futuras.
            exploration_rate: Probabilidad de exploración (0-1), controla cuánto explora el agente.
        """
        self.action_space = action_space  # Lista de acciones posibles.
        self.learning_rate = learning_rate  # Tasa de aprendizaje.
        self.discount_factor = discount_factor  # Factor de descuento para recompensas futuras.
        self.exploration_rate = exploration_rate  # Probabilidad de explorar en lugar de explotar.
        # Diccionario para almacenar los valores Q, inicializados en 0 para cada acción.
        self.q_values = defaultdict(lambda: np.zeros(len(action_space)))
        self.last_state = None  # Último estado visitado por el agente.
        self.last_action = None  # Última acción tomada por el agente.

    def choose_action(self, state):
        """
        Elige una acción usando la política ε-greedy.
        
        Args:
            state: Estado actual del agente.
        
        Returns:
            action: Acción seleccionada.
        """
        state_key = self._state_to_key(state)  # Convertir el estado a una clave hashable.
        
        if random.random() < self.exploration_rate:  # Condición para explorar.
            # Exploración: elige una acción aleatoria.
            action = random.choice(self.action_space)
        else:
            # Explotación: elige la mejor acción según los valores Q actuales.
            action_idx = np.argmax(self.q_values[state_key])  # Índice de la acción con mayor valor Q.
            action = self.action_space[action_idx]  # Acción correspondiente al índice.
        
        # Guardar el estado y la acción para la actualización posterior.
        self.last_state = state_key
        self.last_action = action
        return action

    def update(self, next_state, reward):
        """
        Actualiza los valores Q usando la regla de Q-learning.
        
        Args:
            next_state: Estado alcanzado después de tomar la acción.
            reward: Recompensa recibida por la acción tomada.
        """
        if self.last_state is None:  # Si no hay un estado previo, no se puede actualizar.
            return
            
        next_state_key = self._state_to_key(next_state)  # Convertir el siguiente estado a clave hashable.
        current_q = self.q_values[self.last_state][self.action_space.index(self.last_action)]  # Valor Q actual.
        
        # Calcular el máximo valor Q para el siguiente estado.
        max_next_q = np.max(self.q_values[next_state_key])
        
        # Actualizar el valor Q usando la fórmula de Q-learning.
        new_q = current_q + self.learning_rate * (
            reward + self.discount_factor * max_next_q - current_q
        )
        
        # Guardar el nuevo valor Q.
        self.q_values[self.last_state][self.action_space.index(self.last_action)] = new_q

    def _state_to_key(self, state):
        """
        Convierte un estado a una clave hashable para usar en el diccionario de valores Q.
        
        Args:
            state: Estado actual.
        
        Returns:
            str: Clave hashable del estado.
        """
        return str(state)  # Convertir el estado a string para usarlo como clave.

# ------------------------------------------
# EJEMPLO: AGENTE PARA LABERINTO ONLINE
# ------------------------------------------

class LaberintoOnline:
    def __init__(self, size=5):
        """
        Inicializa un laberinto cuadrado de tamaño `size x size`.
        
        Args:
            size: Tamaño del laberinto.
        """
        self.size = size  # Tamaño del laberinto.
        self.goal = (size-1, size-1)  # La meta está en la esquina inferior derecha.
        self.reset()  # Reiniciar el laberinto.
        
    def reset(self):
        """
        Reinicia el laberinto y coloca al agente en la posición inicial.
        
        Returns:
            tuple: Posición inicial del agente.
        """
        self.position = (0, 0)  # Posición inicial en la esquina superior izquierda.
        return self.position
    
    def step(self, action):
        """
        Ejecuta una acción y devuelve el nuevo estado, la recompensa y si se alcanzó la meta.
        
        Args:
            action: Acción a ejecutar ('arriba', 'abajo', 'izquierda', 'derecha').
        
        Returns:
            tuple: (nuevo_estado, recompensa, terminado).
        """
        x, y = self.position  # Coordenadas actuales del agente.
        
        # Mover según la acción.
        if action == 'arriba' and x > 0:  # Mover hacia arriba si no está en el borde superior.
            x -= 1
        elif action == 'abajo' and x < self.size-1:  # Mover hacia abajo si no está en el borde inferior.
            x += 1
        elif action == 'izquierda' and y > 0:  # Mover hacia la izquierda si no está en el borde izquierdo.
            y -= 1
        elif action == 'derecha' and y < self.size-1:  # Mover hacia la derecha si no está en el borde derecho.
            y += 1
            
        self.position = (x, y)  # Actualizar la posición del agente.
        
        # Calcular recompensa.
        reward = 1 if self.position == self.goal else -0.1  # Recompensa positiva al alcanzar la meta.
        done = self.position == self.goal  # Verificar si se alcanzó la meta.
        
        return self.position, reward, done

if __name__ == "__main__":
    # Configuración del agente y el entorno.
    acciones = ['arriba', 'abajo', 'izquierda', 'derecha']  # Espacio de acciones.
    agente = OnlineSearchAgent(acciones, learning_rate=0.1, exploration_rate=0.3)  # Crear agente.
    entorno = LaberintoOnline(size=5)  # Crear entorno del laberinto.
    
    # Entrenamiento del agente.
    print("=== ENTRENAMIENTO ===")
    for episodio in range(100):  # Entrenar durante 100 episodios.
        estado = entorno.reset()  # Reiniciar el laberinto.
        total_recompensa = 0  # Acumular recompensa total.
        pasos = 0  # Contar pasos en el episodio.
        
        while True:
            accion = agente.choose_action(estado)  # Elegir acción.
            nuevo_estado, recompensa, terminado = entorno.step(accion)  # Ejecutar acción.
            agente.update(nuevo_estado, recompensa)  # Actualizar valores Q.
            
            total_recompensa += recompensa
            pasos += 1
            estado = nuevo_estado
            
            if terminado:  # Si se alcanzó la meta, terminar el episodio.
                break
        
        if (episodio + 1) % 10 == 0:  # Mostrar progreso cada 10 episodios.
            print(f"Episodio {episodio+1}: Recompensa={total_recompensa:.1f}, Pasos={pasos}")
    
    # Prueba final del agente.
    print("\n=== PRUEBA FINAL ===")
    estado = entorno.reset()
    terminado = False
    pasos = 0
    
    while not terminado:
        accion = agente.choose_action(estado)  # Elegir acción según la política aprendida.
        estado, recompensa, terminado = entorno.step(accion)  # Ejecutar acción.
        print(f"Posición: {estado}, Acción: {accion}")
        pasos += 1
    
    print(f"\n¡Meta alcanzada en {pasos} pasos!")