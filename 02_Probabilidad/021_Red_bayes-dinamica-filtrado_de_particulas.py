import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos numéricos.
                    # Es útil para:
                    # - Representar las partículas y sus pesos como arreglos.
                    # - Realizar cálculos eficientes con operaciones vectorizadas.
                    # - Generar ruido aleatorio para simular movimiento y mediciones.

import matplotlib.pyplot as plt  # Librería para la visualización de datos.
                                 # Es útil para:
                                 # - Graficar posiciones reales, mediciones ruidosas y estimaciones del filtro de partículas.
                                 # - Comparar visualmente los resultados del modelo con los datos simulados.
                                 # - Personalizar gráficos (títulos, etiquetas, leyendas, cuadrículas, etc.).

from scipy.stats import norm  # Proporciona herramientas para trabajar con distribuciones estadísticas.
                              # Es útil para:
                              # - Calcular la probabilidad de las partículas dado el valor medido (PDF de la distribución normal).
                              # - Modelar el ruido del sensor como una distribución normal.

class ParticleFilter:
    def __init__(self, num_particles, motion_noise, sensor_noise):
        """
        Inicializa el filtro de partículas.
        
        :param num_particles: Número de partículas
        :param motion_noise: Ruido del movimiento (std)
        :param sensor_noise: Ruido del sensor (std)
        """
        self.num_particles = num_particles  # Número de partículas en el filtro
        self.motion_noise = motion_noise  # Desviación estándar del ruido del movimiento
        self.sensor_noise = sensor_noise  # Desviación estándar del ruido del sensor
        self.particles = np.random.uniform(0, 10, num_particles)  # Posiciones iniciales aleatorias de las partículas
        self.weights = np.ones(num_particles) / num_particles  # Pesos iniciales uniformes para todas las partículas

    def predict(self, velocity):
        """
        Propaga las partículas según el modelo de movimiento.
        
        :param velocity: Velocidad del movimiento
        """
        # Actualiza las posiciones de las partículas añadiendo la velocidad y ruido gaussiano
        self.particles += velocity + np.random.normal(0, self.motion_noise, self.num_particles)

    def update(self, measurement):
        """
        Actualiza los pesos de las partículas basado en la medición del sensor.
        
        :param measurement: Valor medido por el sensor
        """
        # Calcula la probabilidad de cada partícula dado el valor medido
        likelihood = norm.pdf(measurement, loc=self.particles, scale=self.sensor_noise)
        self.weights *= likelihood  # Ajusta los pesos según la probabilidad
        self.weights += 1e-12  # Evita que los pesos sean exactamente cero
        self.weights /= np.sum(self.weights)  # Normaliza los pesos para que sumen 1

    def resample(self):
        """
        Realiza un remuestreo sistemático para evitar la degeneración de partículas.
        """
        # Selecciona partículas basándose en sus pesos
        indices = np.random.choice(
            range(self.num_particles),
            size=self.num_particles,
            p=self.weights,
            replace=True
        )
        self.particles = self.particles[indices]  # Actualiza las partículas seleccionadas
        self.weights = np.ones(self.num_particles) / self.num_particles  # Reinicia los pesos uniformemente

    def estimate(self):
        """
        Calcula la estimación ponderada de la posición.
        
        :return: Estimación de la posición basada en las partículas y sus pesos
        """
        return np.sum(self.particles * self.weights)  # Media ponderada de las partículas

# Parámetros de simulación
np.random.seed(42)  # Fija la semilla para reproducibilidad
true_position = 0.0  # Posición inicial real
velocity = 0.1  # Velocidad constante del movimiento
motion_noise = 0.2  # Ruido del movimiento
sensor_noise = 0.5  # Ruido del sensor
steps = 50  # Número de pasos de simulación

# Inicializar el filtro de partículas
pf = ParticleFilter(
    num_particles=1000,  # Número de partículas
    motion_noise=motion_noise,  # Ruido del movimiento
    sensor_noise=sensor_noise  # Ruido del sensor
)

# Almacenar resultados para análisis
true_positions = []  # Posiciones reales en cada paso
measurements = []  # Mediciones ruidosas del sensor
estimates = []  # Estimaciones del filtro de partículas

# Simulación
for _ in range(steps):
    # Movimiento real (desconocido para el filtro)
    true_position += velocity + np.random.normal(0, motion_noise)  # Actualiza la posición real con ruido
    true_positions.append(true_position)  # Guarda la posición real
    
    # Medición ruidosa del sensor
    z = true_position + np.random.normal(0, sensor_noise)  # Genera una medición ruidosa
    measurements.append(z)  # Guarda la medición
    
    # Filtrado de partículas
    pf.predict(velocity)  # Predice la nueva posición de las partículas
    pf.update(z)  # Actualiza los pesos de las partículas con la medición
    pf.resample()  # Remuestrea las partículas para evitar degeneración
    estimates.append(pf.estimate())  # Calcula y guarda la estimación

# Visualización de resultados
plt.figure(figsize=(12, 6))
plt.plot(true_positions, 'g-', label="Posición real", linewidth=2)  # Traza la posición real
plt.plot(measurements, 'ro', alpha=0.3, label="Mediciones")  # Traza las mediciones ruidosas
plt.plot(estimates, 'b-', label="Estimación PF", linewidth=2)  # Traza las estimaciones del filtro
plt.title("Seguimiento de Posición con Filtro de Partículas")  # Título del gráfico
plt.xlabel("Tiempo")  # Etiqueta del eje X
plt.ylabel("Posición")  # Etiqueta del eje Y
plt.legend()  # Muestra la leyenda
plt.grid(True)  # Activa la cuadrícula
plt.show()  # Muestra el gráfico