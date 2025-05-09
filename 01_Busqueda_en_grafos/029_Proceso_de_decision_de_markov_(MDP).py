# Importa defaultdict para crear diccionarios con valores por defecto
from collections import defaultdict  

class MDP:
    """Clase que implementa un Proceso de Decisión de Markov (MDP)"""
    
    def __init__(self, estados, acciones, transiciones, recompensas, gamma=0.95):
        """
        Inicializa el MDP con sus componentes esenciales.
        
        Args:
            estados: Lista de identificadores de estados (ej. [0, 1, 2])
            acciones: Lista de acciones posibles (ej. ['arriba', 'abajo'])
            transiciones: Diccionario de probabilidades de transición
            recompensas: Diccionario que mapea transiciones a recompensas
            gamma: Factor de descuento para recompensas futuras (0.95 por defecto)
        """
        # Almacena los parámetros como atributos de la clase
        self.estados = estados  
        self.acciones = acciones  
        self.transiciones = transiciones  
        self.recompensas = recompensas  
        self.gamma = gamma  
        # Construye la representación interna del grafo
        self.grafo = self._construir_grafo()  

    def _construir_grafo(self):
        """Construye una representación gráfica del MDP para visualización"""
        # defaultdict crea automáticamente entradas para nuevas claves
        grafo = defaultdict(dict)  
        
        # Itera sobre todas las transiciones definidas
        for (estado_actual, accion), destinos in self.transiciones.items():
            for estado_siguiente, probabilidad in destinos.items():
                # Almacena la probabilidad y recompensa para cada transición
                grafo[(estado_actual, accion)][estado_siguiente] = {
                    'probabilidad': probabilidad,
                    # Usa get() para evitar KeyError, devuelve 0 si no existe
                    'recompensa': self.recompensas.get((estado_actual, accion, estado_siguiente), 0)  
                }
        return grafo

    def iteracion_valores(self, epsilon=1e-6, max_iter=1000):
        """
        Implementa el algoritmo de Iteración de Valores para resolver el MDP.
        
        Args:
            epsilon: Umbral de convergencia (diferencia mínima entre iteraciones)
            max_iter: Número máximo de iteraciones permitidas
            
        Returns:
            tuple: (valores_optimos, politica_optima)
        """
        # Paso 1: Inicialización - Valores a 0 para todos los estados
        valores = {estado: 0 for estado in self.estados}  
        
        for _ in range(max_iter):  # Loop principal del algoritmo
            delta = 0  # Para rastrear el mayor cambio en esta iteración
            nuevos_valores = valores.copy()  # Copia para actualización simultánea
            
            for estado in self.estados:  # Paso 2: Actualización de valores
                # Saltar estados terminales (sin transiciones salientes)
                if estado not in [s for (s, a) in self.transiciones.keys()]:
                    continue
                
                # Diccionario para almacenar valores Q(s,a)
                valores_accion = {}  
                
                for accion in self.acciones:  # Calcula Q(s,a) para cada acción
                    if (estado, accion) not in self.transiciones:
                        continue  # Saltar acciones no definidas
                        
                    # Inicializa el valor Q para esta acción
                    valor_q = 0  
                    
                    # Calcula el valor esperado usando la ecuación de Bellman
                    for estado_sig, prob in self.transiciones[(estado, accion)].items():
                        recompensa = self.recompensas.get((estado, accion, estado_sig), 0)
                        valor_q += prob * (recompensa + self.gamma * valores[estado_sig])
                    
                    valores_accion[accion] = valor_q  # Almacena Q(s,a)
                
                # Actualiza V(s) con el máximo Q(s,a) si hay acciones válidas
                if valores_accion:
                    nuevos_valores[estado] = max(valores_accion.values())
                    # Actualiza delta con el mayor cambio absoluto
                    delta = max(delta, abs(nuevos_valores[estado] - valores[estado]))  
            
            valores = nuevos_valores  # Actualiza los valores para la siguiente iteración
            
            # Criterio de convergencia: si el cambio máximo es menor que epsilon
            if delta < epsilon:  
                break

        # Paso 3: Extraer política óptima
        politica = {}
        
        for estado in self.estados:
            # Estados terminales no tienen acción asociada
            if estado not in [s for (s, a) in self.transiciones.keys()]:
                politica[estado] = None
                continue
                
            valores_accion = {}  # Nuevamente calculamos Q(s,a) para cada acción
            
            for accion in self.acciones:
                if (estado, accion) not in self.transiciones:
                    continue
                    
                valor_q = 0
                for estado_sig, prob in self.transiciones[(estado, accion)].items():
                    recompensa = self.recompensas.get((estado, accion, estado_sig), 0)
                    valor_q += prob * (recompensa + self.gamma * valores[estado_sig])
                
                valores_accion[accion] = valor_q
            
            # La política es la acción con máximo Q(s,a)
            politica[estado] = max(valores_accion, key=valores_accion.get) if valores_accion else None
        
        return valores, politica

    def visualizar_grafo(self):
        """Muestra una representación legible del grafo de transiciones"""
        print("\nGrafo del MDP (Estado, Acción) -> [Destinos]:")
        for (estado, accion), destinos in self.grafo.items():
            print(f"\nDesde ({estado}, {accion}):")
            for estado_sig, datos in destinos.items():
                print(f"  → {estado_sig} [Probabilidad={datos['probabilidad']:.2f}, Recompensa={datos['recompensa']}]")

# Función de ejemplo: Problema de gestión de inventario
def ejemplo_inventario():
    """
    Configura y resuelve un MDP para un problema simple de gestión de inventario.
    
    Returns:
        tuple: (instancia_mdp, valores_optimos, politica_optima)
    """
    # Definición de estados (niveles de inventario)
    estados = [0, 1, 2]  
    
    # Acciones posibles (unidades a pedir)
    acciones = [0, 1, 2]  
    
    # Probabilidades de transición (dinámica del sistema)
    transiciones = {
        (0, 0): {0: 0.5, 1: 0.3, 2: 0.2},
        (0, 1): {1: 0.5, 2: 0.3, 0: 0.2},
        (0, 2): {2: 0.5, 1: 0.3, 0: 0.2},
        (1, 0): {0: 0.6, 1: 0.3, 2: 0.1},
        (1, 1): {1: 0.6, 2: 0.3, 0: 0.1},
        (1, 2): {2: 0.6, 1: 0.3, 0: 0.1},
        (2, 0): {1: 0.7, 2: 0.2, 0: 0.1},
        (2, 1): {2: 0.7, 1: 0.2, 0: 0.1},
        (2, 2): {2: 0.8, 1: 0.1, 0: 0.1}
    }
    
    # Función de recompensa (costos/beneficios)
    recompensas = {
        (0, 0, 0): -10,  # Costo por falta de stock
        (0, 1, 1): -2,   # Costo de almacenamiento
        (0, 2, 2): -4,
        (1, 0, 0): 5,    # Beneficio por venta
        (1, 1, 1): 3,
        (2, 0, 1): 8,
        (2, 2, 2): 2
    }
    
    # Crear instancia del MDP con gamma=0.9
    mdp = MDP(estados, acciones, transiciones, recompensas, gamma=0.9)
    
    # Mostrar la estructura del grafo
    mdp.visualizar_grafo()
    
    # Resolver el MDP usando Iteración de Valores
    valores, politica = mdp.iteracion_valores()
    
    # Mostrar resultados
    print("\nValores óptimos por estado:")
    for estado in estados:
        print(f"Estado {estado}: {valores[estado]:.2f}")
    
    print("\nPolítica óptima:")
    for estado in estados:
        print(f"En estado {estado}: pedir {politica[estado]} unidades")
    
    return mdp, valores, politica

# Punto de entrada principal
if __name__ == "__main__":
    # Ejecutar el ejemplo cuando se corre el script directamente
    ejemplo_inventario()