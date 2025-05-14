import numpy as np  # Para realizar operaciones matemáticas avanzadas y manejo eficiente de arreglos multidimensionales.
import matplotlib.pyplot as plt  # Para crear gráficos y visualizar datos, como trayectorias y estimaciones.
from pykalman import KalmanFilter  # Para implementar y aplicar el Filtro de Kalman en datos ruidosos.

# 1. Datos simulados: Generación de la trayectoria real y observaciones ruidosas
np.random.seed(42)  # Fijar la semilla para reproducibilidad
tiempo = np.linspace(0, 10, 100)  # Vector de tiempo (100 puntos entre 0 y 10)
# Trayectoria real: posición en X aumenta linealmente, en Y sigue una onda sinusoidal
posicion_real = np.column_stack((0.5 * tiempo, 2 * np.sin(tiempo)))  
# Ruido gaussiano: simulación de errores en los sensores
ruido = np.random.normal(0, 0.5, posicion_real.shape)                
# Observaciones ruidosas: combinación de la trayectoria real con ruido
observaciones = posicion_real + ruido                                

# 2. Configurar Filtro de Kalman
kf = KalmanFilter(
    initial_state_mean=observaciones[0],  # Estimación inicial: primera observación ruidosa
    n_dim_obs=2,                          # Dimensiones observadas (x, y)
    transition_matrices=np.array([[1, 0], [0, 1]]),  # Modelo de movimiento: constante
    observation_matrices=np.array([[1, 0], [0, 1]]), # Relación directa entre estado y observación
    transition_covariance=0.1 * np.eye(2),           # Incertidumbre en el modelo de movimiento
    observation_covariance=0.5 * np.eye(2)           # Incertidumbre en las observaciones
)

# 3. Filtrar las observaciones ruidosas
# El método `smooth` aplica el filtro de Kalman para suavizar las observaciones
estados_suavizados, _ = kf.smooth(observaciones)

# 4. Visualización de los resultados
plt.figure(figsize=(12, 6))  # Configurar el tamaño de la figura
# Graficar las observaciones ruidosas
plt.plot(observaciones[:, 0], observaciones[:, 1], 'ro', alpha=0.5, label='Observaciones (ruidosas)')
# Graficar la trayectoria real
plt.plot(posicion_real[:, 0], posicion_real[:, 1], 'k-', lw=2, label='Trayectoria real')
# Graficar la estimación del filtro de Kalman
plt.plot(estados_suavizados[:, 0], estados_suavizados[:, 1], 'b-', lw=2, label='Filtro de Kalman (estimación)')
# Configurar leyenda, título y etiquetas
plt.legend()
plt.title('Seguimiento de Movimiento con Filtro de Kalman')
plt.xlabel('Posición X')
plt.ylabel('Posición Y')
plt.grid()  # Mostrar cuadrícula
plt.show()  # Mostrar la gráfica