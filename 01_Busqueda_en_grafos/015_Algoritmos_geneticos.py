import random  # Se utiliza para generar números aleatorios, lo cual es esencial para:
               # - Inicializar la población de manera aleatoria.
               # - Seleccionar individuos para cruce y mutación.
               # - Generar factores aleatorios en el cruce y mutación.

import numpy as np  # Proporciona herramientas para operaciones matemáticas avanzadas y manejo eficiente de arreglos.
                    # Es útil para:
                    # - Crear y manipular vectores (individuos).
                    # - Realizar cálculos matemáticos como suma, producto, etc.
                    # - Aplicar restricciones a los valores de los genes (np.clip).

from operator import itemgetter  # Permite acceder a elementos específicos de tuplas o listas.
                                 # Es utilizado para:
                                 # - Seleccionar el mejor individuo basado en su fitness.
                                 # - Ordenar o filtrar individuos según su valor de fitness.

from math import sin, cos  # Proporciona funciones trigonométricas como seno y coseno.
                           # Estas funciones son útiles para definir funciones objetivo complejas
                           # que dependen de relaciones trigonométricas.

class AlgoritmoGenetico:
    def __init__(self, funcion_objetivo, tamano_poblacion=50, 
                 prob_cruce=0.8, prob_mutacion=0.1, elitismo=True):
        """
        Inicializa el algoritmo genético con los parámetros dados.
        
        Args:
            funcion_objetivo: Función a maximizar (debe recibir un individuo).
            tamano_poblacion: Número de individuos por generación.
            prob_cruce: Probabilidad de cruce (0-1).
            prob_mutacion: Probabilidad de mutación por gen (0-1).
            elitismo: Si True, conserva el mejor individuo entre generaciones.
        """
        self.funcion = funcion_objetivo  # Función objetivo a maximizar
        self.tamano_poblacion = tamano_poblacion  # Tamaño de la población
        self.prob_cruce = prob_cruce  # Probabilidad de cruce
        self.prob_mutacion = prob_mutacion  # Probabilidad de mutación
        self.elitismo = elitismo  # Habilitar o deshabilitar elitismo

    def _inicializar_individuo(self, espacio_busqueda, longitud_genotipo):
        """
        Crea un individuo aleatorio dentro del espacio de búsqueda.
        
        Args:
            espacio_busqueda: Tupla (min, max) con los límites del espacio.
            longitud_genotipo: Número de genes por individuo.
        
        Returns:
            np.ndarray: Individuo generado aleatoriamente.
        """
        # Genera un vector aleatorio dentro de los límites especificados
        return np.random.uniform(espacio_busqueda[0], espacio_busqueda[1], longitud_genotipo)

    def _evaluar_poblacion(self, poblacion):
        """
        Evalúa todos los individuos de la población calculando su fitness.
        
        Args:
            poblacion: Lista de individuos.
        
        Returns:
            list: Lista de tuplas (individuo, fitness).
        """
        # Calcula el fitness de cada individuo usando la función objetivo
        return [(ind, self.funcion(ind)) for ind in poblacion]

    def _seleccion_por_torneo(self, poblacion_evaluada, tamano_torneo=3):
        """
        Selección por torneo entre 'tamano_torneo' individuos aleatorios.
        
        Args:
            poblacion_evaluada: Lista de tuplas (individuo, fitness).
            tamano_torneo: Número de participantes en cada torneo.
        
        Returns:
            list: Lista de individuos seleccionados.
        """
        seleccionados = []
        for _ in range(self.tamano_poblacion):
            # Selecciona aleatoriamente participantes para el torneo
            participantes = random.sample(poblacion_evaluada, tamano_torneo)
            # Elige al ganador del torneo (mayor fitness)
            ganador = max(participantes, key=itemgetter(1))[0]
            seleccionados.append(ganador)
        return seleccionados

    def _cruzar(self, padre1, padre2):
        """
        Realiza un cruce aritmético entre dos padres.
        
        Args:
            padre1: Primer padre.
            padre2: Segundo padre.
        
        Returns:
            np.ndarray: Hijo generado por cruce.
        """
        # Genera un factor de mezcla aleatorio
        alpha = random.random()
        # Combina los genes de los padres usando el factor de mezcla
        return alpha * padre1 + (1 - alpha) * padre2

    def _mutar(self, individuo, espacio_busqueda, fuerza_mutacion=0.1):
        """
        Aplica mutación gaussiana a cada gen con probabilidad prob_mutacion.
        
        Args:
            individuo: Individuo a mutar.
            espacio_busqueda: Tupla (min, max) con los límites del espacio.
            fuerza_mutacion: Desviación estándar de la mutación.
        
        Returns:
            np.ndarray: Individuo mutado.
        """
        for i in range(len(individuo)):
            # Decide si mutar el gen actual
            if random.random() < self.prob_mutacion:
                # Aplica una mutación gaussiana al gen
                individuo[i] += random.gauss(0, fuerza_mutacion)
                # Asegura que el gen permanezca dentro de los límites
                individuo[i] = np.clip(individuo[i], espacio_busqueda[0], espacio_busqueda[1])
        return individuo

    def ejecutar(self, espacio_busqueda, longitud_genotipo, generaciones=100, verbose=True):
        """
        Ejecuta el algoritmo genético completo.
        
        Args:
            espacio_busqueda: Tupla (min, max) con los límites del espacio.
            longitud_genotipo: Número de genes por individuo.
            generaciones: Número máximo de generaciones.
            verbose: Si True, muestra progreso.
        
        Returns:
            tuple: (mejor_individuo, mejor_fitness, historial_fitness).
        """
        # 1. Inicialización de la población
        poblacion = [self._inicializar_individuo(espacio_busqueda, longitud_genotipo) 
                     for _ in range(self.tamano_poblacion)]
        
        historial_fitness = []  # Historial de los mejores fitness por generación
        mejor_individuo = None  # Mejor individuo encontrado
        mejor_fitness = -np.inf  # Mejor fitness encontrado (inicializado a -infinito)

        for gen in range(generaciones):
            # 2. Evaluación de la población
            poblacion_evaluada = self._evaluar_poblacion(poblacion)
            
            # Actualizar el mejor individuo global
            mejor_actual = max(poblacion_evaluada, key=itemgetter(1))
            if mejor_actual[1] > mejor_fitness:
                mejor_individuo, mejor_fitness = mejor_actual
            
            # Registrar el mejor fitness de esta generación
            historial_fitness.append(mejor_fitness)
            
            # Mostrar progreso cada 10 generaciones si verbose está activado
            if verbose and gen % 10 == 0:
                print(f"Gen {gen}: Mejor fitness = {mejor_fitness:.4f}")

            # 3. Selección de individuos para la siguiente generación
            seleccionados = self._seleccion_por_torneo(poblacion_evaluada)
            
            # 4. Reproducción para crear la nueva generación
            nueva_generacion = []
            
            # Elitismo: conservar el mejor individuo
            if self.elitismo:
                nueva_generacion.append(mejor_individuo.copy())
            
            # Generar nuevos individuos mediante cruce y mutación
            while len(nueva_generacion) < self.tamano_poblacion:
                # Seleccionar dos padres aleatoriamente
                padre1, padre2 = random.sample(seleccionados, 2)
                
                # Realizar cruce con probabilidad prob_cruce
                if random.random() < self.prob_cruce:
                    hijo = self._cruzar(padre1, padre2)
                else:
                    # Si no hay cruce, copiar uno de los padres
                    hijo = padre1.copy() if random.random() < 0.5 else padre2.copy()
                
                # Aplicar mutación al hijo
                hijo = self._mutar(hijo, espacio_busqueda)
                
                # Añadir el hijo a la nueva generación
                nueva_generacion.append(hijo)
            
            # Reemplazar la población actual con la nueva generación
            poblacion = nueva_generacion
        
        # Retornar el mejor individuo, su fitness y el historial de fitness
        return mejor_individuo, mejor_fitness, historial_fitness

# ------------------------------------------
# EJEMPLO DE USO: OPTIMIZACIÓN DE FUNCIÓN
# ------------------------------------------

def funcion_objetivo(x):
    """
    Función multimodal de ejemplo con múltiples máximos.
    
    Args:
        x: np.ndarray con los valores de los genes.
    
    Returns:
        float: Valor de la función objetivo.
    """
    # Calcula el valor de la función objetivo para un vector x
    return sin(x[0]) * cos(x[1]) * (1 / (1 + abs(x[2]))) + 0.1 * np.sum(x**2)

if __name__ == "__main__":
    # Configuración del algoritmo genético
    ag = AlgoritmoGenetico(
        funcion_objetivo=funcion_objetivo,
        tamano_poblacion=100,
        prob_cruce=0.9,
        prob_mutacion=0.05,
        elitismo=True
    )
    
    # Ejecutar la optimización
    mejor_sol, mejor_valor, historial = ag.ejecutar(
        espacio_busqueda=(-5, 5),
        longitud_genotipo=3,  # Dimensión del problema
        generaciones=200,
        verbose=True
    )
    
    # Mostrar resultados finales
    print("\n=== RESULTADOS FINALES ===")
    print(f"Mejor solución encontrada: {mejor_sol}")
    print(f"Valor de la función: {mejor_valor:.6f}")
    print(f"Progreso a través de generaciones:")
    for gen in range(0, len(historial), 20):
        print(f"Gen {gen:3d}: {historial[gen]:.4f}")