import numpy as np
import matplotlib.pyplot as plt

class KalmanFilter:
    def __init__(self, initial_state, initial_uncertainty, F, H, Q, R):
        """
        Inicialización del Filtro de Kalman.
        
        Parámetros:
        - initial_state: Estado inicial [posición, velocidad].
        - initial_uncertainty: Matriz de covarianza inicial.
        - F: Matriz de transición de estado (modelo dinámico).
        - H: Matriz de observación (qué se mide).
        - Q: Ruido del proceso (incertidumbre en el modelo).
        - R: Ruido de medición (incertidumbre en los sensores).
        """
        self.x = initial_state  # Estado inicial
        self.P = initial_uncertainty  # Incertidumbre inicial
        self.F = F  # Matriz de transición de estado
        self.H = H  # Matriz de observación
        self.Q = Q  # Ruido del proceso
        self.R = R  # Ruido de medición
        self.history_states = []  # Historial de estados para suavizado
        self.history_covariances = []  # Historial de covarianzas

    def predict(self):
        """Predice el siguiente estado basado en el modelo dinámico."""
        self.x = np.dot(self.F, self.x)  # Predicción del estado
        self.P = np.dot(np.dot(self.F, self.P), self.F.T) + self.Q  # Predicción de la incertidumbre
        return self.x

    def update(self, measurement):
        """
        Actualiza el estado con una nueva medición.
        
        Parámetros:
        - measurement: Nueva medición observada.
        """
        y = measurement - np.dot(self.H, self.x)  # Residuo (diferencia entre medición y predicción)
        S = np.dot(np.dot(self.H, self.P), self.H.T) + self.R  # Incertidumbre de la medición
        K = np.dot(np.dot(self.P, self.H.T), np.linalg.inv(S))  # Ganancia de Kalman
        self.x = self.x + np.dot(K, y)  # Actualización del estado
        self.P = self.P - np.dot(np.dot(K, self.H), self.P)  # Actualización de la incertidumbre
        self.history_states.append(self.x.copy())  # Guardar estado actualizado
        self.history_covariances.append(self.P.copy())  # Guardar covarianza actualizada
        return self.x

    def smooth(self):
        """
        Suavizado de Rauch-Tung-Striebel para mejorar estimaciones pasadas.
        """
        n = len(self.history_states)  # Número de estados en el historial
        smoothed_states = [self.history_states[-1]]  # Iniciar con el último estado
        smoothed_covs = [self.history_covariances[-1]]  # Iniciar con la última covarianza

        # Iterar hacia atrás para suavizar
        for t in range(n-2, -1, -1):
            P_pred = np.dot(np.dot(self.F, self.history_covariances[t]), self.F.T) + self.Q  # Predicción de covarianza
            G = np.dot(np.dot(self.history_covariances[t], self.F.T), np.linalg.inv(P_pred))  # Ganancia de suavizado
            x_smooth = self.history_states[t] + np.dot(G, smoothed_states[0] - np.dot(self.F, self.history_states[t]))
            P_smooth = self.history_covariances[t] + np.dot(np.dot(G, smoothed_covs[0] - P_pred), G.T)
            smoothed_states.insert(0, x_smooth)  # Insertar estado suavizado
            smoothed_covs.insert(0, P_smooth)  # Insertar covarianza suavizada

        return smoothed_states

    def forecast(self, steps):
        """
        Predice estados futuros sin mediciones.
        
        Parámetros:
        - steps: Número de pasos a predecir.
        """
        forecasted = []  # Lista para guardar predicciones
        x_temp, P_temp = self.x.copy(), self.P.copy()  # Copias temporales del estado y la incertidumbre

        for _ in range(steps):
            x_temp = np.dot(self.F, x_temp)  # Predicción del estado futuro
            P_temp = np.dot(np.dot(self.F, P_temp), self.F.T) + self.Q  # Predicción de la incertidumbre futura
            forecasted.append(x_temp.copy())  # Guardar predicción

        return forecasted

# --- Configuración del modelo ---
dt = 1.0  # Intervalo de tiempo entre mediciones
F = np.array([[1, dt], [0, 1]])  # Modelo de movimiento (velocidad constante)
H = np.array([[1, 0]])  # Solo medimos la posición
Q = np.array([[0.01, 0], [0, 0.01]])  # Ruido del proceso (pequeño)
R = np.array([[0.5]])  # Ruido de medición (alto)

# --- Simulación ---
np.random.seed(42)  # Fijar semilla para reproducibilidad
true_positions = np.linspace(0, 10, 20)  # Movimiento lineal real
noisy_measurements = true_positions + np.random.normal(0, np.sqrt(R[0,0]), 20)  # Mediciones ruidosas

# --- Inicialización del filtro ---
kf = KalmanFilter(
    initial_state=np.array([0, 0.5]),  # [posición inicial, velocidad inicial]
    initial_uncertainty=np.eye(2),  # Matriz de covarianza inicial
    F=F, H=H, Q=Q, R=R
)

# --- Filtrado en tiempo real ---
filtered_states = []  # Lista para guardar estados filtrados
for z in noisy_measurements:
    kf.predict()  # Predicción del siguiente estado
    filtered_states.append(kf.update(z)[0])  # Actualización con medición y guardar posición

# --- Suavizado de la trayectoria ---
smoothed_states = [x[0] for x in kf.smooth()]  # Extraer posiciones suavizadas

# --- Predicción a futuro ---
future_steps = 5  # Número de pasos a predecir
forecasted = [x[0] for x in kf.forecast(future_steps)]  # Predicciones futuras de posición

# --- Visualización ---
plt.figure(figsize=(10, 6))
plt.plot(true_positions, 'g-', label="Posición real")  # Trayectoria real
plt.plot(noisy_measurements, 'ro', label="Mediciones ruidosas")  # Mediciones con ruido
plt.plot(filtered_states, 'b-', label="Filtrado de Kalman")  # Estados filtrados
plt.plot(smoothed_states, 'm--', label="Suavizado")  # Estados suavizados
plt.plot(range(20, 20+future_steps), forecasted, 'c-', label="Predicción futura")  # Predicciones futuras
plt.legend()
plt.title("Filtrado, Predicción, Suavizado y Explicación con Kalman")
plt.xlabel("Tiempo")
plt.ylabel("Posición")
plt.grid(True)
plt.show()