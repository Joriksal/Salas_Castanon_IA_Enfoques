# Importamos defaultdict de la librería collections
# defaultdict es una subclase de diccionario que permite inicializar valores por defecto
from collections import defaultdict

# Definimos la clase BayesianNetwork, que representa una red bayesiana
class BayesianNetwork:
    def __init__(self):
        """
        Constructor de la clase BayesianNetwork.
        Aquí se define la estructura de la red bayesiana y las probabilidades condicionales (CPTs).
        """
        # Estructura de la red: nodos y sus padres
        # Cada nodo es una variable aleatoria, y los padres son las variables que afectan su probabilidad
        self.structure = {
            'Batería': [],  # Nodo raíz sin padres
            'Alternador': ['Batería'],  # Nodo con un padre: Batería
            'Bujías': [],  # Nodo raíz sin padres
            'Motor': ['Batería', 'Alternador', 'Bujías'],  # Nodo con múltiples padres
            'Encendido': ['Motor'],  # Nodo con un padre: Motor
            'LuzTablero': ['Batería', 'Alternador'],  # Nodo con dos padres
            'Arranque': ['Batería', 'Motor']  # Nodo con dos padres
        }
        
        # Probabilidades condicionales (CPTs - Conditional Probability Tables)
        # Estas tablas definen las probabilidades de cada nodo dado el estado de sus padres
        self.cpts = {
            'Batería': {'T': 0.95, 'F': 0.05},  # Probabilidad de que la batería funcione ('T') o no ('F')
            'Bujías': {'T': 0.90, 'F': 0.10},  # Probabilidad de que las bujías funcionen o no
            'Alternador': {  # Probabilidad del alternador dado el estado de la batería
                'T': {'T': 0.80, 'F': 0.20},  # Si la batería funciona, el alternador tiene 80% de funcionar
                'F': {'T': 0.10, 'F': 0.90}   # Si la batería no funciona, el alternador tiene 10% de funcionar
            },
            'Motor': {  # Probabilidad del motor dado el estado de la batería, alternador y bujías
                ('T', 'T', 'T'): {'T': 0.99, 'F': 0.01},  # Todas las entradas funcionan
                ('T', 'T', 'F'): {'T': 0.10, 'F': 0.90},  # Una entrada falla, etc.
                ('T', 'F', 'T'): {'T': 0.85, 'F': 0.15},
                ('T', 'F', 'F'): {'T': 0.05, 'F': 0.95},
                ('F', 'T', 'T'): {'T': 0.70, 'F': 0.30},
                ('F', 'T', 'F'): {'T': 0.01, 'F': 0.99},
                ('F', 'F', 'T'): {'T': 0.15, 'F': 0.85},
                ('F', 'F', 'F'): {'T': 0.001, 'F': 0.999}  # Todas las entradas fallan
            },
            'Encendido': {  # Probabilidad del encendido dado el estado del motor
                'T': {'T': 0.95, 'F': 0.05},  # Motor funcionando
                'F': {'T': 0.02, 'F': 0.98}   # Motor no funcionando
            },
            'LuzTablero': {  # Probabilidad de la luz del tablero dado el estado de la batería y alternador
                ('T', 'T'): {'T': 0.02, 'F': 0.98},  # Ambos funcionan
                ('T', 'F'): {'T': 0.95, 'F': 0.05},  # Solo batería funciona
                ('F', 'T'): {'T': 0.10, 'F': 0.90},  # Solo alternador funciona
                ('F', 'F'): {'T': 0.99, 'F': 0.01}   # Ninguno funciona
            },
            'Arranque': {  # Probabilidad del arranque dado el estado de la batería y motor
                ('T', 'T'): {'T': 0.97, 'F': 0.03},  # Ambos funcionan
                ('T', 'F'): {'T': 0.60, 'F': 0.40},  # Solo batería funciona
                ('F', 'T'): {'T': 0.05, 'F': 0.95},  # Solo motor funciona
                ('F', 'F'): {'T': 0.001, 'F': 0.999} # Ninguno funciona
            }
        }
    
    def get_prob(self, node, parent_values, value):
        """
        Obtiene la probabilidad condicional P(node=value|parents=parent_values).
        - node: Nodo actual.
        - parent_values: Valores de los padres del nodo.
        - value: Valor del nodo ('T' o 'F').
        """
        if not self.structure[node]:  # Si el nodo no tiene padres (es raíz)
            return self.cpts[node][value]
        
        if len(self.structure[node]) == 1:  # Si el nodo tiene un solo padre
            return self.cpts[node][parent_values[0]][value]
        else:  # Si el nodo tiene múltiples padres
            return self.cpts[node][parent_values][value]
    
    def infer(self, evidence):
        """
        Realiza inferencia por enumeración para calcular P(X|evidence).
        - evidence: Diccionario con las variables observadas y sus valores.
        """
        # Identificamos las variables ocultas (no incluidas en la evidencia)
        hidden_vars = [var for var in self.structure if var not in evidence]
        
        # Inicializamos la distribución posterior como un diccionario con valores por defecto 0
        posterior = defaultdict(float)
        
        # Enumeramos todas las posibles combinaciones de valores para las variables ocultas
        from itertools import product  # Importamos product para generar combinaciones
        possible_values = ['T', 'F']  # Valores posibles para cada variable
        for values in product(possible_values, repeat=len(hidden_vars)):
            # Creamos una asignación completa combinando evidencia y valores ocultos
            current_assignment = evidence.copy()
            current_assignment.update(zip(hidden_vars, values))
            
            # Calculamos la probabilidad conjunta para esta asignación
            joint_prob = 1.0
            for node in self.structure:
                parents = self.structure[node]  # Obtenemos los padres del nodo
                parent_values = tuple(current_assignment[p] for p in parents)  # Valores de los padres
                node_value = current_assignment[node]  # Valor del nodo actual
                joint_prob *= self.get_prob(node, parent_values, node_value)  # Multiplicamos las probabilidades
            
            # Sumamos la probabilidad conjunta a la distribución posterior
            for var in hidden_vars:
                if current_assignment[var] == 'T':  # Solo nos interesa cuando la variable es 'T'
                    posterior[var] += joint_prob
        
        # Normalizamos la distribución posterior para que las probabilidades sumen 1
        total = sum(posterior.values())
        if total > 0:
            for var in posterior:
                posterior[var] /= total
        
        return posterior

# Ejemplo de uso de la red bayesiana
if __name__ == "__main__":
    # Creamos una instancia de la red bayesiana
    bn = BayesianNetwork()
    
    # Caso 1: No arranca y luz del tablero encendida
    print("=== Caso 1: No arranca y luz del tablero encendida ===")
    evidence_1 = {'Arranque': 'F', 'LuzTablero': 'T'}  # Evidencia observada
    result_1 = bn.infer(evidence_1)  # Realizamos inferencia
    for var, prob in result_1.items():  # Iteramos sobre los resultados
        print(f"P({var}=T|evidencia) = {prob:.4f}")  # Mostramos las probabilidades
    
    # Caso 2: No arranca pero sin luz del tablero
    print("\n=== Caso 2: No arranca pero sin luz del tablero ===")
    evidence_2 = {'Arranque': 'F', 'LuzTablero': 'F'}
    result_2 = bn.infer(evidence_2)
    for var, prob in result_2.items():
        print(f"P({var}=T|evidencia) = {prob:.4f}")
    
    # Caso 3: Problemas de encendido
    print("\n=== Caso 3: Problemas de encendido ===")
    evidence_3 = {'Encendido': 'F'}
    result_3 = bn.infer(evidence_3)
    for var, prob in result_3.items():
        print(f"P({var}=T|evidencia) = {prob:.4f}")