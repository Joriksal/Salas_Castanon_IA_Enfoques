# Importaciones necesarias
import numpy as np  # Librería para operaciones matemáticas avanzadas.
from itertools import product  # Para generar combinaciones de decisiones posibles.
from collections import defaultdict  # Para crear un diccionario con valores por defecto como listas.

class RedDecision:
    def __init__(self):
        """
        Inicializa una red de decisión.
        - `grafo`: Representa las dependencias entre nodos como un grafo dirigido.
        - `nodos`: Contiene información sobre cada nodo (tipo, opciones, probabilidades, etc.).
        - `utilidades`: Almacena las tablas de utilidad asociadas a combinaciones de nodos padres.
        """
        self.grafo = defaultdict(list)  # Grafo dirigido que almacena dependencias entre nodos.
        self.nodos = {}  # Diccionario para almacenar información de los nodos.
        self.utilidades = {}  # Diccionario para almacenar tablas de utilidad.
    
    def agregar_nodo(self, nombre, tipo, **kwargs):
        """
        Agrega un nodo a la red de decisión.
        
        Args:
            nombre: Nombre del nodo.
            tipo: Tipo del nodo ('chance', 'decision').
            kwargs: Información adicional como opciones o probabilidades.
        """
        self.nodos[nombre] = {'tipo': tipo, **kwargs}  # Almacena el nodo con su tipo y atributos adicionales.
    
    def agregar_arista(self, origen, destino):
        """
        Agrega una dependencia entre nodos en el grafo.
        
        Args:
            origen: Nodo padre.
            destino: Nodo hijo.
        """
        self.grafo[origen].append(destino)  # Añade el nodo hijo al nodo padre en el grafo.
    
    def asignar_probabilidad(self, nodo, probabilidades):
        """
        Asigna una tabla de probabilidades a un nodo de tipo 'chance'.
        
        Args:
            nodo: Nombre del nodo.
            probabilidades: Diccionario con las probabilidades de los valores del nodo.
        """
        self.nodos[nodo]['probabilidades'] = probabilidades  # Asocia las probabilidades al nodo.
    
    def asignar_utilidad(self, padres, tabla):
        """
        Define la función de utilidad para un conjunto de nodos padres.
        
        Args:
            padres: Lista de nodos padres.
            tabla: Diccionario que asocia combinaciones de valores de los padres con utilidades.
        """
        self.utilidades[tuple(padres)] = tabla  # Almacena la tabla de utilidad asociada a los padres.
    
    def mejor_decision(self, evidencias={}):
        """
        Encuentra la mejor decisión dada la evidencia actual.
        
        Args:
            evidencias: Diccionario con valores observados para algunos nodos.
        
        Returns:
            tuple: (Mejor combinación de decisiones, utilidad esperada).
        """
        # Identificar nodos de decisión que no están en las evidencias.
        nodos_decision = [n for n, attr in self.nodos.items() 
                         if attr['tipo'] == 'decision' and n not in evidencias]
        
        # Generar todas las combinaciones posibles de decisiones.
        decisiones_posibles = product(*[self.nodos[n]['opciones'] for n in nodos_decision])
        
        mejor_utilidad = -np.inf  # Inicializar la mejor utilidad como un valor muy bajo.
        mejor_combinacion = None  # Inicializar la mejor combinación de decisiones.
        
        for combo in decisiones_posibles:
            # Crear un diccionario con la combinación actual de decisiones.
            decision = dict(zip(nodos_decision, combo))
            
            # Calcular la utilidad total para esta combinación de decisiones.
            utilidad = self._calcular_utilidad({**evidencias, **decision})
            
            # Actualizar la mejor combinación si la utilidad es mayor.
            if utilidad > mejor_utilidad:
                mejor_utilidad = utilidad
                mejor_combinacion = decision
        
        return mejor_combinacion, mejor_utilidad  # Retorna la mejor combinación y su utilidad.
    
    def _calcular_utilidad(self, estados):
        """
        Calcula la utilidad total para un conjunto de estados.
        
        Args:
            estados: Diccionario con valores asignados a los nodos.
        
        Returns:
            float: Utilidad total calculada.
        """
        utilidad = 0  # Inicializar la utilidad total.
        for padres, tabla in self.utilidades.items():
            # Obtener los valores de los padres según los estados actuales.
            valores = tuple(estados[p] for p in padres if p in estados)
            
            # Sumar la utilidad correspondiente si los valores están en la tabla.
            if valores in tabla:
                utilidad += tabla[valores]
        return utilidad  # Retorna la utilidad total calculada.

class ValorInformacion:
    def __init__(self, red):
        """
        Inicializa el calculador de Valor de la Información (VOI).
        
        Args:
            red: Objeto de la clase RedDecision.
        """
        self.red = red  # Asocia la red de decisión al calculador de VOI.
    
    def calcular_voi(self, nodo_info, evidencias={}):
        """
        Calcula el Valor de la Información (VOI) para un nodo.
        
        Args:
            nodo_info: Nombre del nodo de tipo 'chance'.
            evidencias: Diccionario con valores observados para algunos nodos.
        
        Returns:
            float: Valor de la Información (VOI).
        """
        # Utilidad esperada sin información adicional.
        _, util_sin_info = self.red.mejor_decision(evidencias)
        
        # Utilidad esperada con información perfecta.
        util_con_info = 0
        opciones = self.red.nodos[nodo_info]['opciones']
        
        for valor in opciones:
            # Probabilidad marginal P(nodo_info=valor).
            prob = self.red.nodos[nodo_info]['probabilidades'].get(valor, 0)
            if prob > 0:
                # Calcular la utilidad esperada dado nodo_info=valor.
                nueva_evidencia = {**evidencias, nodo_info: valor}
                _, util = self.red.mejor_decision(nueva_evidencia)
                util_con_info += prob * util
        
        # El VOI es la diferencia entre la utilidad con y sin información.
        return max(0, util_con_info - util_sin_info)

def crear_ejemplo_medico():
    """
    Crea y configura una red de decisión médica de ejemplo.
    
    Returns:
        RedDecision: Red de decisión configurada.
    """
    red = RedDecision()
    
    # Nodos chance (incertidumbre).
    red.agregar_nodo('Enfermedad', 'chance', 
                    opciones=['presente', 'ausente'],
                    probabilidades={'presente': 0.2, 'ausente': 0.8})
    
    red.agregar_nodo('Test', 'chance',
                    opciones=['positivo', 'negativo'],
                    probabilidades={
                        'positivo': 0.26,  # P(Test=positivo) marginal.
                        'negativo': 0.74
                    },
                    costo=50)
    
    # Nodos decisión.
    red.agregar_nodo('Tratamiento', 'decision',
                    opciones=['medicar', 'observar', 'cirugia'])
    
    # Dependencias.
    red.agregar_arista('Enfermedad', 'Test')
    
    # Función de utilidad.
    red.asignar_utilidad(
        ['Enfermedad', 'Tratamiento'],
        {
            ('presente', 'medicar'): 80,
            ('presente', 'observar'): -50,
            ('presente', 'cirugia'): 60,
            ('ausente', 'medicar'): -30,
            ('ausente', 'observar'): 100,
            ('ausente', 'cirugia'): -80
        }
    )
    
    return red

if __name__ == "__main__":
    # Crear y analizar la red.
    red_medica = crear_ejemplo_medico()
    voi_calculator = ValorInformacion(red_medica)
    
    # Calcular VOI para el test diagnóstico.
    voi = voi_calculator.calcular_voi('Test')
    costo_test = red_medica.nodos['Test']['costo']
    
    print(f"\nValor de la Información para el Test: ${voi:.2f}")
    print(f"Costo del Test: ${costo_test:.2f}")
    print(f"Recomendación: {'Realizar test' if voi > costo_test else 'No realizar test'}")
    
    # Mostrar mejor decisión sin información adicional.
    decision, utilidad = red_medica.mejor_decision()
    print(f"\nMejor decisión sin test: {decision} (Utilidad esperada: ${utilidad:.2f})")