import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos numéricos.
                    # Es útil para:
                    # - Representar estados, matrices de transición y covarianzas en el Filtro de Kalman.
                    # - Realizar cálculos matriciales como multiplicaciones y transposiciones.
                    # - Generar ruido aleatorio para simulaciones de datos reales y mediciones.

import matplotlib.pyplot as plt  # Librería para la visualización de datos.
                                 # Es útil para:
                                 # - Graficar posiciones reales, mediciones ruidosas y estimaciones del Filtro de Kalman.
                                 # - Comparar visualmente los resultados del modelo con los datos simulados.
                                 # - Personalizar gráficos (títulos, etiquetas, leyendas, cuadrículas, etc.).

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, F, H, Q, R):
        """
        Inicializa el filtro de Kalman.
        
        :param initial_state: Vector de estado inicial [posición, velocidad]
        :param initial_uncertainty: Matriz de covarianza inicial
        :param F: Matriz de transición de estado
        :param H: Matriz de observación
        :param Q: Matriz de ruido del proceso
        :param R: Matriz de ruido de medición
        """
        self.x = initial_state  # Estado inicial
        self.P = initial_uncertainty  # Incertidumbre inicial
        self.F = F  # Matriz de transición de estado
        self.H = H  # Matriz de observación
        self.Q = Q  # Ruido del proceso
        self.R = R  # Ruido de medición
        self.history = []  # Historial de estados estimados

    def predict(self):
        """
        Predice el siguiente estado basado en el modelo.
        """
        self.x = self.F @ self.x  # Predicción del estado
        self.P = self.F @ self.P @ self.F.T + self.Q  # Actualización de la incertidumbre
        return self.x

    def update(self, measurement):
        """
        Actualiza el estado con una nueva medición.
        
        :param measurement: Valor medido (posición en este caso)
        """
        y = measurement - self.H @ self.x  # Residual (diferencia entre medición y predicción)
        S = self.H @ self.P @ self.H.T + self.R  # Covarianza residual
        K = self.P @ self.H.T @ np.linalg.inv(S)  # Ganancia de Kalman (peso de la medición)
        
        self.x = self.x + K @ y  # Actualización del estado
        self.P = (np.eye(len(self.x)) - K @ self.H) @ self.P  # Actualización de la incertidumbre
        
        self.history.append(self.x.copy())  # Guardar el estado estimado
        return self.x

# Configuración del sistema
dt = 0.1  # Intervalo de tiempo entre mediciones
F = np.array([[1, dt], 
              [0, 1]])  # Modelo de movimiento con velocidad constante
H = np.array([[1, 0]])  # Solo medimos la posición
Q = np.diag([0.01, 0.001])  # Ruido del proceso (incertidumbre en el modelo)
R = np.array([[0.25]])  # Ruido de medición (incertidumbre del GPS)

# Simulación de datos reales y mediciones
np.random.seed(42)  # Fijar la semilla para reproducibilidad
true_states = []  # Lista para almacenar las posiciones reales
measurements = []  # Lista para almacenar las mediciones simuladas
true_state = np.array([0, 0.5])  # Estado inicial real [posición, velocidad]

for _ in range(100):
    # Actualización del estado real con ruido del proceso
    true_state = F @ true_state + np.random.multivariate_normal([0, 0], Q)
    true_states.append(true_state[0])  # Guardar la posición real
    # Generar medición simulada con ruido de medición
    measurements.append(H @ true_state + np.random.normal(0, np.sqrt(R[0, 0])))

# Filtrado con Kalman
kf = KalmanFilter(
    initial_state=np.array([0, 0]),  # Estado inicial estimado
    initial_uncertainty=np.eye(2),  # Incertidumbre inicial (matriz identidad)
    F=F, H=H, Q=Q, R=R  # Parámetros del modelo
)

estimates = []  # Lista para almacenar las estimaciones del filtro de Kalman
for z in measurements:
    kf.predict()  # Predicción del siguiente estado
    estimates.append(kf.update(z)[0])  # Actualización con la medición y guardar la posición estimada

# Visualización de resultados
plt.figure(figsize=(12, 6))
plt.plot(true_states, 'g-', label="Posición real", linewidth=2)  # Posición real
plt.plot(measurements, 'ro', alpha=0.5, label="Mediciones GPS")  # Mediciones simuladas
plt.plot(estimates, 'b--', label="Estimación Kalman", linewidth=2)  # Estimaciones del filtro de Kalman
plt.title("Seguimiento de Posición con Filtro de Kalman")
plt.xlabel("Tiempo (pasos)")
plt.ylabel("Posición (m)")
plt.legend()
plt.grid(True)
plt.show()