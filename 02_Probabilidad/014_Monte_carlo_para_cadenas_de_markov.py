import numpy as np  # Librería para operaciones matemáticas y generación de números aleatorios.
                    # Es útil para:
                    # - Generar muestras aleatorias de distribuciones como normal y uniforme.
                    # - Realizar cálculos matemáticos avanzados y operaciones vectorizadas.
                    # - Calcular estadísticas como media, desviación estándar y autocorrelación.

import matplotlib.pyplot as plt  # Librería para la visualización de datos.
                                 # Es útil para:
                                 # - Crear gráficos como histogramas y trayectorias de cadenas de Markov.
                                 # - Comparar las muestras generadas con la distribución objetivo.
                                 # - Personalizar visualizaciones (títulos, etiquetas, cuadrículas, etc.).

from scipy.stats import norm, uniform  # Proporciona herramientas para trabajar con distribuciones estadísticas.
                                       # Es útil para:
                                       # - Definir distribuciones como normal (norm) y uniforme (uniform).
                                       # - Calcular funciones de densidad de probabilidad (PDF) y generar valores aleatorios.

# Definición de la distribución objetivo
def target_distribution(x):
    """
    Distribución objetivo (mezcla de dos Gaussianas).
    Esta es la distribución de la cual queremos generar muestras.
    """
    return 0.6 * norm.pdf(x, loc=2, scale=1) + 0.4 * norm.pdf(x, loc=-1, scale=0.5)

# Implementación del algoritmo Metropolis-Hastings
def metropolis_hastings(target, n_samples, x_init=0, proposal_std=1, burn_in=1000):
    """
    Algoritmo Metropolis-Hastings para generar muestras de una distribución objetivo.

    Args:
        target: Función de distribución objetivo.
        n_samples: Número de muestras a generar después del periodo de burn-in.
        x_init: Valor inicial de la cadena.
        proposal_std: Desviación estándar de la distribución de propuesta (normal).
        burn_in: Número de muestras iniciales a descartar (para estabilizar la cadena).

    Returns:
        Muestras generadas de la distribución objetivo.
    """
    samples = []  # Lista para almacenar las muestras
    x_current = x_init  # Inicialización de la cadena
    accepted = 0  # Contador de propuestas aceptadas

    # Iterar para generar muestras
    for i in range(n_samples + burn_in):
        # Generar una propuesta desde una distribución normal centrada en x_current
        x_proposed = np.random.normal(x_current, proposal_std)
        
        # Calcular el ratio de aceptación (target(x_proposed) / target(x_current))
        acceptance_ratio = target(x_proposed) / target(x_current)
        
        # Aceptar o rechazar la propuesta con probabilidad igual al ratio de aceptación
        if np.random.uniform(0, 1) < acceptance_ratio:
            x_current = x_proposed  # Actualizar el estado actual
            if i >= burn_in:  # Contar solo después del periodo de burn-in
                accepted += 1
        
        # Guardar la muestra después del periodo de burn-in
        if i >= burn_in:
            samples.append(x_current)
    
    # Calcular y mostrar la tasa de aceptación
    acceptance_rate = accepted / n_samples
    print(f"Tasa de aceptación: {acceptance_rate:.2%}")
    return np.array(samples)  # Retornar las muestras como un arreglo de NumPy

# Parámetros del algoritmo
n_samples = 10000  # Número de muestras a generar después del burn-in
burn_in = 2000  # Número de muestras iniciales a descartar

# Ejecutar el algoritmo Metropolis-Hastings
samples = metropolis_hastings(target_distribution, n_samples, burn_in=burn_in)

# Visualización de los resultados
plt.figure(figsize=(12, 6))

# Histograma de las muestras generadas
plt.subplot(1, 2, 1)
plt.hist(samples, bins=50, density=True, alpha=0.7, color='skyblue')  # Histograma de las muestras
x_vals = np.linspace(-4, 5, 1000)  # Valores para graficar la distribución objetivo
plt.plot(x_vals, target_distribution(x_vals), 'r-', lw=2)  # Distribución objetivo
plt.title('Muestras MCMC vs Distribución Objetivo')
plt.xlabel('x')
plt.ylabel('Densidad')

# Trayectoria de la cadena (primeras 500 muestras)
plt.subplot(1, 2, 2)
plt.plot(samples[:500], 'b-', alpha=0.5)  # Graficar las primeras 500 muestras
plt.title('Trayectoria de la Cadena (primeras 500 muestras)')
plt.xlabel('Iteración')
plt.ylabel('Valor de x')
plt.grid(True)

# Ajustar el diseño de la figura
plt.tight_layout()
plt.show()

# Estadísticas de convergencia
print("\nESTADÍSTICAS DE CONVERGENCIA:")
print(f"Media muestral: {np.mean(samples):.4f}")  # Media de las muestras
print(f"Desviación estándar: {np.std(samples):.4f}")  # Desviación estándar de las muestras
print(f"Autocorrelación (lag 1): {np.corrcoef(samples[:-1], samples[1:])[0,1]:.4f}")  # Autocorrelación lag-1