# Importamos las librerías necesarias
import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos numéricos
import matplotlib.pyplot as plt  # Librería para crear gráficos y visualizaciones

# Verificar estilos disponibles y configurar
# `plt.style.available` devuelve una lista de estilos predefinidos en matplotlib
print("Estilos disponibles en matplotlib:", plt.style.available)
plt.style.use('ggplot')  # Configuramos el estilo de los gráficos a 'ggplot' para una mejor estética

# Intentar importar librerías adicionales para análisis estadístico
try:
    # `statsmodels` es una librería para análisis estadístico avanzado
    from statsmodels.tsa.stattools import adfuller  # Prueba de Dickey-Fuller para verificar estacionariedad
    from statsmodels.graphics.tsaplots import plot_acf  # Gráfico de autocorrelación
    statsmodels_available = True  # Indicamos que la librería está disponible
except ImportError:
    # Si no se puede importar, mostramos un mensaje de advertencia
    statsmodels_available = False
    print("Advertencia: statsmodels no está instalado. Algunas funciones estarán limitadas.")
    from scipy import signal  # Usamos scipy como alternativa básica para análisis de señales

# Mensaje para instalar statsmodels si no está disponible
if not statsmodels_available:
    print("\nPara un análisis completo, instala statsmodels ejecutando:")
    print("pip install statsmodels")  # Instrucción para instalar la librería faltante

# =============================================
# 1. GENERACIÓN DE PROCESOS ESTACIONARIOS
# =============================================

def generate_ar1_process(n=1000, phi=0.8, sigma=1):
    """
    Genera un proceso AR(1) estacionario.
    Parámetros:
        n: Número de puntos en la serie (tamaño de la serie temporal).
        phi: Coeficiente de autoregresión (controla la dependencia entre valores consecutivos).
        sigma: Desviación estándar del ruido blanco (aleatoriedad en el modelo).
    Retorna:
        Serie temporal generada como un arreglo de numpy.
    """
    x = np.zeros(n)  # Inicializar la serie con ceros (arreglo de tamaño n)
    for t in range(1, n):  # Iteramos desde el segundo elemento (t=1) hasta el final
        # Fórmula del modelo AR(1): x[t] = phi * x[t-1] + ruido blanco
        x[t] = phi * x[t-1] + np.random.normal(0, sigma)  # np.random.normal genera ruido blanco
    return x  # Retornamos la serie generada

def generate_random_walk(n=1000):
    """
    Genera un camino aleatorio (no estacionario).
    Parámetros:
        n: Número de puntos en la serie.
    Retorna:
        Serie temporal generada como un camino aleatorio.
    """
    # np.cumsum calcula la suma acumulativa de un arreglo, simulando un camino aleatorio
    return np.cumsum(np.random.normal(0, 1, n))  # Generamos ruido blanco y calculamos su suma acumulativa

# =============================================
# 2. FUNCIONES DE ANÁLISIS MEJORADAS
# =============================================

def test_stationarity(series, title):
    """
    Realiza un análisis de estacionariedad en una serie temporal.
    Parámetros:
        series: Serie temporal a analizar (arreglo de numpy).
        title: Título descriptivo de la serie (para gráficos y resultados).
    """
    # Crear una figura con dos subgráficos
    plt.figure(figsize=(14, 5))  # Tamaño de la figura en pulgadas (ancho x alto)
    
    # Gráfico de la serie temporal
    plt.subplot(1, 2, 1)  # Primer subgráfico (1 fila, 2 columnas, posición 1)
    plt.plot(series, color='steelblue')  # Graficamos la serie con color azul acero
    plt.title(f'Serie Temporal: {title}\n(Media: {np.mean(series):.2f}, Desv: {np.std(series):.2f})')
    plt.grid(True, alpha=0.3)  # Agregamos una cuadrícula con transparencia

    # Histograma y densidad
    plt.subplot(1, 2, 2)  # Segundo subgráfico (posición 2)
    plt.hist(series, bins=30, density=True, color='skyblue', alpha=0.7)  # Histograma con 30 bins
    plt.title('Distribución de Valores')
    plt.grid(True, alpha=0.3)  # Cuadrícula

    plt.tight_layout()  # Ajustar automáticamente los subgráficos para evitar solapamientos
    plt.show()  # Mostrar los gráficos

    # Análisis estadístico solo si statsmodels está disponible
    if statsmodels_available:
        result = adfuller(series)  # Prueba de Dickey-Fuller para verificar estacionariedad
        print(f"\nAnálisis estadístico para {title}:")
        print(f'ADF Statistic: {result[0]:.4f}')  # Estadístico de la prueba
        print(f'p-value: {result[1]:.4f}')  # Valor p (probabilidad de rechazar la hipótesis nula)
        print('Valores críticos:')  # Valores críticos para diferentes niveles de significancia
        for key, value in result[4].items():
            print(f'   {key}: {value:.4f}')
        
        # Conclusión basada en el valor p
        if result[1] <= 0.05:  # Si el valor p es menor o igual a 0.05, rechazamos la hipótesis nula
            print("Conclusión: Serie ESTACIONARIA (rechazamos H0)")
        else:
            print("Conclusión: Serie NO ESTACIONARIA (no podemos rechazar H0)")
        
        # Gráfico de autocorrelación
        plt.figure(figsize=(10, 4))  # Tamaño del gráfico
        plot_acf(series, lags=40, color='teal')  # Gráfico de autocorrelación con 40 retardos
        plt.title(f'Función de Autocorrelación: {title}')
        plt.grid(True, alpha=0.3)  # Cuadrícula
        plt.show()
    else:
        # Mensaje si statsmodels no está disponible
        print("\nInstala statsmodels para análisis completo:")
        print("pip install statsmodels")

# =============================================
# 3. EJEMPLO PRÁCTICO
# =============================================

# Generar datos
ar1_stationary = generate_ar1_process(phi=0.7)  # Proceso AR(1) estacionario con phi=0.7
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
        diff: Orden de la diferenciación (por defecto, 1).
    Retorna:
        Serie transformada como un arreglo de numpy.
    """
    stationary_series = np.diff(series, n=diff)  # np.diff calcula la diferencia entre valores consecutivos
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
        series_list: Lista de series temporales (arreglos de numpy).
        titles: Lista de títulos para cada serie.
    """
    n = len(series_list)  # Número de series a comparar
    rows = (n + 1) // 2  # Calcular dinámicamente el número de filas (2 series por fila)
    cols = 2 if n > 1 else 1  # Usar 2 columnas si hay más de una serie
    
    plt.figure(figsize=(15, 4 * rows))  # Tamaño de la figura
    for i, (series, title) in enumerate(zip(series_list, titles), 1):  # Iterar sobre series y títulos
        plt.subplot(rows, cols, i)  # Crear subgráficos
        plt.plot(series, color=f'C{i}')  # Graficar cada serie con un color diferente
        plt.title(title)  # Título del subgráfico
        plt.grid(True, alpha=0.3)  # Cuadrícula
    
    plt.tight_layout()  # Ajustar automáticamente los subgráficos
    plt.show()  # Mostrar la figura

# Comparar procesos
compare_series([ar1_stationary, random_walk], 
               ["AR(1) Estacionario", "Camino Aleatorio"])