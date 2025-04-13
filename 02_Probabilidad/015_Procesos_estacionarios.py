import numpy as np
import matplotlib.pyplot as plt

# Verificar estilos disponibles y configurar
print("Estilos disponibles en matplotlib:", plt.style.available)
plt.style.use('ggplot')  # Configuramos el estilo de los gráficos a 'ggplot'

# Intentar importar librerías adicionales para análisis estadístico
try:
    from statsmodels.tsa.stattools import adfuller  # Prueba de Dickey-Fuller para estacionariedad
    from statsmodels.graphics.tsaplots import plot_acf  # Gráfico de autocorrelación
    statsmodels_available = True
except ImportError:
    statsmodels_available = False
    print("Advertencia: statsmodels no está instalado. Algunas funciones estarán limitadas.")
    from scipy import signal  # Alternativa básica para autocorrelación

# Mensaje para instalar statsmodels si no está disponible
if not statsmodels_available:
    print("\nPara un análisis completo, instala statsmodels ejecutando:")
    print("pip install statsmodels")

# =============================================
# 1. GENERACIÓN DE PROCESOS ESTACIONARIOS
# =============================================

def generate_ar1_process(n=1000, phi=0.8, sigma=1):
    """
    Genera un proceso AR(1) estacionario.
    Parámetros:
        n: Número de puntos en la serie.
        phi: Coeficiente de autoregresión.
        sigma: Desviación estándar del ruido blanco.
    Retorna:
        Serie temporal generada.
    """
    x = np.zeros(n)  # Inicializar la serie con ceros
    for t in range(1, n):
        x[t] = phi * x[t-1] + np.random.normal(0, sigma)  # Fórmula AR(1)
    return x

def generate_random_walk(n=1000):
    """
    Genera un camino aleatorio (no estacionario).
    Parámetros:
        n: Número de puntos en la serie.
    Retorna:
        Serie temporal generada.
    """
    return np.cumsum(np.random.normal(0, 1, n))  # Suma acumulativa de ruido blanco

# =============================================
# 2. FUNCIONES DE ANÁLISIS MEJORADAS
# =============================================

def test_stationarity(series, title):
    """
    Realiza un análisis de estacionariedad en una serie temporal.
    Parámetros:
        series: Serie temporal a analizar.
        title: Título descriptivo de la serie.
    """
    plt.figure(figsize=(14, 5))
    
    # Gráfico de la serie temporal
    plt.subplot(1, 2, 1)
    plt.plot(series, color='steelblue')
    plt.title(f'Serie Temporal: {title}\n(Media: {np.mean(series):.2f}, Desv: {np.std(series):.2f})')
    plt.grid(True, alpha=0.3)
    
    # Histograma y densidad
    plt.subplot(1, 2, 2)
    plt.hist(series, bins=30, density=True, color='skyblue', alpha=0.7)
    plt.title('Distribución de Valores')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Análisis estadístico solo si statsmodels está disponible
    if statsmodels_available:
        result = adfuller(series)  # Prueba de Dickey-Fuller
        print(f"\nAnálisis estadístico para {title}:")
        print(f'ADF Statistic: {result[0]:.4f}')  # Estadístico de la prueba
        print(f'p-value: {result[1]:.4f}')  # Valor p
        print('Valores críticos:')
        for key, value in result[4].items():
            print(f'   {key}: {value:.4f}')
        
        # Conclusión basada en el valor p
        if result[1] <= 0.05:
            print("Conclusión: Serie ESTACIONARIA (rechazamos H0)")
        else:
            print("Conclusión: Serie NO ESTACIONARIA (no podemos rechazar H0)")
        
        # Gráfico de autocorrelación
        plt.figure(figsize=(10, 4))
        plot_acf(series, lags=40, color='teal')
        plt.title(f'Función de Autocorrelación: {title}')
        plt.grid(True, alpha=0.3)
        plt.show()
    else:
        print("\nInstala statsmodels para análisis completo:")
        print("pip install statsmodels")

# =============================================
# 3. EJEMPLO PRÁCTICO
# =============================================

# Generar datos
ar1_stationary = generate_ar1_process(phi=0.7)  # Proceso AR(1) estacionario
random_walk = generate_random_walk()  # Camino aleatorio no estacionario

# Analizar series
test_stationarity(ar1_stationary, "AR(1) Estacionario (φ=0.7)")
test_stationarity(random_walk, "Camino Aleatorio")

# =============================================
# 4. TRANSFORMACIÓN A ESTACIONARIEDAD
# =============================================

def make_stationary(series, diff=1):
    """
    Transforma una serie no estacionaria en estacionaria mediante diferenciación.
    Parámetros:
        series: Serie temporal a transformar.
        diff: Orden de la diferenciación.
    Retorna:
        Serie transformada.
    """
    stationary_series = np.diff(series, n=diff)  # Diferenciación de la serie
    print(f"Serie transformada a estacionaria. Longitud reducida de {len(series)} a {len(stationary_series)}.")
    return stationary_series

# Ejemplo de transformación
if statsmodels_available:
    rw_stationary = make_stationary(random_walk)  # Diferenciar el camino aleatorio
    test_stationarity(rw_stationary, "Camino Aleatorio Diferenciado")

# =============================================
# 5. VISUALIZACIÓN COMPARATIVA
# =============================================

def compare_series(series_list, titles):
    """
    Compara múltiples series en una sola figura.
    Parámetros:
        series_list: Lista de series temporales.
        titles: Lista de títulos para cada serie.
    """
    n = len(series_list)  # Número de series
    rows = (n + 1) // 2  # Calcular filas dinámicamente
    cols = 2 if n > 1 else 1  # Usar 2 columnas si hay más de una serie
    
    plt.figure(figsize=(15, 4 * rows))  # Tamaño de la figura
    for i, (series, title) in enumerate(zip(series_list, titles), 1):
        plt.subplot(rows, cols, i)  # Crear subgráficos
        plt.plot(series, color=f'C{i}')  # Graficar cada serie
        plt.title(title)
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

# Comparar procesos
compare_series([ar1_stationary, random_walk], 
               ["AR(1) Estacionario", "Camino Aleatorio"])