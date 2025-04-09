import numpy as np
from collections import defaultdict

class BayesianNetwork:
    def __init__(self):
        # Estructura de la red: nodos y sus padres
        # Cada nodo tiene una lista de padres que afectan su probabilidad
        self.structure = {
            'Batería': [],  # Nodo raíz sin padres
            'Alternador': ['Batería'],  # Nodo con un padre: Batería
            'Bujías': [],  # Nodo raíz sin padres
            'Motor': ['Batería', 'Alternador', 'Bujías'],  # Nodo con múltiples padres
            'Encendido': ['Motor'],  # Nodo con un padre: Motor
            'LuzTablero': ['Batería', 'Alternador'],  # Nodo con dos padres
            'Arranque': ['Batería', 'Motor']  # Nodo con dos padres
        }
        
        # Probabilidades condicionales (CPTs)
        # Define las probabilidades de cada nodo dado el estado de sus padres
        self.cpts = {
            'Batería': {'T': 0.95, 'F': 0.05},  # Probabilidad de que la batería funcione o no
            'Bujías': {'T': 0.90, 'F': 0.10},  # Probabilidad de que las bujías funcionen o no
            'Alternador': {  # Probabilidad del alternador dado el estado de la batería
                'T': {'T': 0.80, 'F': 0.20},
                'F': {'T': 0.10, 'F': 0.90}
            },
            'Motor': {  # Probabilidad del motor dado el estado de la batería, alternador y bujías
                ('T', 'T', 'T'): {'T': 0.99, 'F': 0.01},
                ('T', 'T', 'F'): {'T': 0.10, 'F': 0.90},
                ('T', 'F', 'T'): {'T': 0.85, 'F': 0.15},
                ('T', 'F', 'F'): {'T': 0.05, 'F': 0.95},
                ('F', 'T', 'T'): {'T': 0.70, 'F': 0.30},
                ('F', 'T', 'F'): {'T': 0.01, 'F': 0.99},
                ('F', 'F', 'T'): {'T': 0.15, 'F': 0.85},
                ('F', 'F', 'F'): {'T': 0.001, 'F': 0.999}
            },
            'Encendido': {  # Probabilidad del encendido dado el estado del motor
                'T': {'T': 0.95, 'F': 0.05},
                'F': {'T': 0.02, 'F': 0.98}
            },
            'LuzTablero': {  # Probabilidad de la luz del tablero dado el estado de la batería y alternador
                ('T', 'T'): {'T': 0.02, 'F': 0.98},
                ('T', 'F'): {'T': 0.95, 'F': 0.05},
                ('F', 'T'): {'T': 0.10, 'F': 0.90},
                ('F', 'F'): {'T': 0.99, 'F': 0.01}
            },
            'Arranque': {  # Probabilidad del arranque dado el estado de la batería y motor
                ('T', 'T'): {'T': 0.97, 'F': 0.03},
                ('T', 'F'): {'T': 0.60, 'F': 0.40},
                ('F', 'T'): {'T': 0.05, 'F': 0.95},
                ('F', 'F'): {'T': 0.001, 'F': 0.999}
            }
        }
    
    def get_prob(self, node, parent_values, value):
        """
        Obtiene la probabilidad condicional P(node=value|parents=parent_values)
        """
        if not self.structure[node]:  # Nodo raíz (sin padres)
            return self.cpts[node][value]
        
        if len(self.structure[node]) == 1:  # Nodo con un solo padre
            return self.cpts[node][parent_values[0]][value]
        else:  # Nodo con múltiples padres
            return self.cpts[node][parent_values][value]
    
    def infer(self, evidence):
        """
        Realiza inferencia por enumeración para calcular P(X|evidence)
        """
        # Variables no observadas (no incluidas en la evidencia)
        hidden_vars = [var for var in self.structure if var not in evidence]
        
        # Inicializar distribución posterior
        posterior = defaultdict(float)
        
        # Enumerar todas las posibles combinaciones de valores para las variables ocultas
        from itertools import product
        possible_values = ['T', 'F']
        for values in product(possible_values, repeat=len(hidden_vars)):
            # Crear una asignación completa combinando evidencia y valores ocultos
            current_assignment = evidence.copy()
            current_assignment.update(zip(hidden_vars, values))
            
            # Calcular la probabilidad conjunta para esta asignación
            joint_prob = 1.0
            for node in self.structure:
                parents = self.structure[node]
                parent_values = tuple(current_assignment[p] for p in parents)
                node_value = current_assignment[node]
                joint_prob *= self.get_prob(node, parent_values, node_value)
            
            # Sumar la probabilidad conjunta a la distribución posterior
            for var in hidden_vars:
                if current_assignment[var] == 'T':  # Solo nos interesa cuando la variable es 'T'
                    posterior[var] += joint_prob
        
        # Normalizar la distribución posterior
        total = sum(posterior.values())
        if total > 0:
            for var in posterior:
                posterior[var] /= total
        
        return posterior

# Ejemplo de uso
if __name__ == "__main__":
    bn = BayesianNetwork()
    
    # Caso 1: No arranca y luz del tablero encendida
    print("=== Caso 1: No arranca y luz del tablero encendida ===")
    evidence_1 = {'Arranque': 'F', 'LuzTablero': 'T'}
    result_1 = bn.infer(evidence_1)
    for var, prob in result_1.items():
        print(f"P({var}=T|evidencia) = {prob:.4f}")
    
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