import numpy as np  # Librería para operaciones matemáticas y generación de números aleatorios.
                    # Es útil para:
                    # - Generar muestras aleatorias de distribuciones como Gamma y Exponencial.
                    # - Realizar cálculos matemáticos avanzados y operaciones vectorizadas.
                    # - Evaluar condiciones y realizar comparaciones numéricas.

import matplotlib.pyplot as plt  # Librería para la visualización de datos.
                                 # Es útil para:
                                 # - Crear gráficos como histogramas y curvas de densidad.
                                 # - Configurar y personalizar visualizaciones (títulos, etiquetas, cuadrículas, etc.).
                                 # - Comparar distribuciones generadas con funciones teóricas.

from scipy.stats import weibull_min  # Proporciona herramientas para trabajar con distribuciones estadísticas.
                                     # Es útil para:
                                     # - Definir la distribución Weibull como objetivo en el muestreo por rechazo.
                                     # - Calcular la función de densidad de probabilidad (PDF) y la función de distribución acumulativa (CDF).

# =============================================
# 1. MUESTREO DIRECTO: Distribución Gamma
# =============================================

# Parámetros de la distribución Gamma
alpha, beta = 3, 2  # alpha: forma, beta: tasa (inverso de escala)
n_samples = 10000   # Número de muestras a generar

# Generación de muestras directas de la distribución Gamma
gamma_samples = np.random.gamma(shape=alpha, scale=1/beta, size=n_samples)

# Visualización de las muestras generadas
plt.figure(figsize=(12, 5))  # Configuración del tamaño de la figura
plt.subplot(1, 2, 1)  # Primera subgráfica
plt.hist(gamma_samples, bins=50, density=True, alpha=0.7, color='skyblue')  # Histograma
plt.title('Muestreo Directo Gamma(3, 2)')  # Título
plt.xlabel('Valor')  # Etiqueta del eje x
plt.ylabel('Densidad')  # Etiqueta del eje y
plt.grid(True)  # Mostrar cuadrícula

# =============================================
# 2. MUESTREO POR RECHAZO: Distribución Weibull
# =============================================

# Parámetros de la distribución Weibull objetivo
k, lam = 2, 1  # k: forma, lam: escala
target_dist = weibull_min(k, scale=lam)  # Distribución objetivo Weibull(2, 1)

# Clase para la distribución propuesta (Exponencial)
class ProposalDistribution:
    def __init__(self, rate):
        self.rate = rate  # Tasa de la distribución exponencial
    
    def sample(self, size):
        # Genera muestras de una distribución exponencial
        return np.random.exponential(scale=1/self.rate, size=size)
    
    def pdf(self, x):
        # Calcula la función de densidad de probabilidad (PDF) de la exponencial
        return self.rate * np.exp(-self.rate * x)

# Parámetros del muestreo por rechazo
M = 2.5  # Constante de escalamiento para garantizar que M*q(x) >= p(x)
q = ProposalDistribution(rate=0.6)  # Distribución propuesta con tasa 0.6

# Función para realizar el muestreo por rechazo
def rejection_sampling(n_samples):
    samples = []  # Lista para almacenar las muestras aceptadas
    n_attempts = 0  # Contador de intentos totales
    
    while len(samples) < n_samples:
        # Generar una muestra de la distribución propuesta
        x = q.sample(1)[0]
        # Generar un número aleatorio uniforme entre 0 y 1
        u = np.random.uniform(0, 1)
        # Calcular la razón de aceptación
        acceptance_ratio = target_dist.pdf(x) / (M * q.pdf(x))
        n_attempts += 1  # Incrementar el contador de intentos
        
        # Aceptar la muestra si u <= razón de aceptación
        if u <= acceptance_ratio:
            samples.append(x)
    
    # Calcular la tasa de aceptación
    acceptance_rate = n_samples / n_attempts
    return np.array(samples), acceptance_rate

# Generación de muestras usando el muestreo por rechazo
weibull_samples, acceptance_rate = rejection_sampling(5000)

# Visualización de los resultados del muestreo por rechazo
plt.subplot(1, 2, 2)  # Segunda subgráfica
x_vals = np.linspace(0, 5, 1000)  # Valores para graficar las funciones
plt.plot(x_vals, target_dist.pdf(x_vals), 'r-', lw=2, label='Weibull(2,1)')  # PDF de Weibull
plt.plot(x_vals, M*q.pdf(x_vals), 'b--', lw=1, label='M*q(x)')  # M*q(x)
plt.hist(weibull_samples, bins=50, density=True, alpha=0.5, color='orange')  # Histograma de muestras
plt.title('Muestreo por Rechazo Weibull')  # Título
plt.xlabel('x')  # Etiqueta del eje x
plt.ylabel('Densidad')  # Etiqueta del eje y
plt.legend()  # Mostrar leyenda
plt.grid(True)  # Mostrar cuadrícula

plt.tight_layout()  # Ajustar el diseño de las gráficas
plt.show()  # Mostrar las gráficas

# =============================================
# Resultados numéricos
# =============================================

# Imprimir resultados del muestreo
print("\nRESULTADOS:")
print(f"Tasa de aceptación: {acceptance_rate:.2%}")  # Porcentaje de muestras aceptadas
print(f"Muestras generadas: {len(weibull_samples)}")  # Número total de muestras generadas
print(f"Estimación P(X > 5): {np.mean(weibull_samples > 5):.6f}")  # Probabilidad estimada de X > 5
print(f"Valor teórico P(X > 5): {1 - target_dist.cdf(5):.6f}")  # Probabilidad teórica de X > 5