import numpy as np
from collections import defaultdict

class ChainRuleModel:
    def __init__(self):
        # Definición de las variables y sus dependencias
        # Cada variable puede depender de otras (sus "padres")
        self.variables = {
            'Background': [],  # No depende de ninguna otra variable
            'Preparation': ['Background'],  # Depende de 'Background'
            'Difficulty': [],  # No depende de ninguna otra variable
            'Effort': ['Preparation', 'Difficulty'],  # Depende de 'Preparation' y 'Difficulty'
            'Skill': ['Effort', 'Background'],  # Depende de 'Effort' y 'Background'
            'Performance': ['Skill', 'Effort', 'Difficulty']  # Depende de 'Skill', 'Effort' y 'Difficulty'
        }
        
        # Tablas de probabilidad condicional (CPTs)
        # Estas tablas definen las probabilidades de cada variable dado el valor de sus padres
        self.cpts = {
            'Background': {'High': 0.3, 'Low': 0.7},  # Probabilidades de 'Background'
            'Preparation': {  # Probabilidades de 'Preparation' dado 'Background'
                'High': {'High': 0.8, 'Low': 0.2},
                'Low': {'High': 0.4, 'Low': 0.6}
            },
            'Difficulty': {'High': 0.4, 'Low': 0.6},  # Probabilidades de 'Difficulty'
            'Effort': {  # Probabilidades de 'Effort' dado 'Preparation' y 'Difficulty'
                ('High', 'High'): {'High': 0.7, 'Low': 0.3},
                ('High', 'Low'): {'High': 0.9, 'Low': 0.1},
                ('Low', 'High'): {'High': 0.5, 'Low': 0.5},
                ('Low', 'Low'): {'High': 0.8, 'Low': 0.2}
            },
            'Skill': {  # Probabilidades de 'Skill' dado 'Effort' y 'Background'
                ('High', 'High'): {'High': 0.95, 'Medium': 0.05, 'Low': 0.0},
                ('High', 'Low'): {'High': 0.7, 'Medium': 0.25, 'Low': 0.05},
                ('Low', 'High'): {'High': 0.5, 'Medium': 0.3, 'Low': 0.2},
                ('Low', 'Low'): {'High': 0.2, 'Medium': 0.4, 'Low': 0.4}
            },
            'Performance': {  # Probabilidades de 'Performance' dado 'Skill', 'Effort' y 'Difficulty'
                ('High', 'High', 'High'): {'A': 0.9, 'B': 0.08, 'C': 0.02},
                ('High', 'High', 'Low'): {'A': 0.95, 'B': 0.04, 'C': 0.01},
                ('High', 'Low', 'High'): {'A': 0.7, 'B': 0.2, 'C': 0.1},
                ('High', 'Low', 'Low'): {'A': 0.8, 'B': 0.15, 'C': 0.05},
                ('Medium', 'High', 'High'): {'A': 0.5, 'B': 0.3, 'C': 0.2},
                ('Medium', 'High', 'Low'): {'A': 0.6, 'B': 0.3, 'C': 0.1},
                ('Medium', 'Low', 'High'): {'A': 0.3, 'B': 0.4, 'C': 0.3},
                ('Medium', 'Low', 'Low'): {'A': 0.4, 'B': 0.4, 'C': 0.2},
                ('Low', 'High', 'High'): {'A': 0.2, 'B': 0.3, 'C': 0.5},
                ('Low', 'High', 'Low'): {'A': 0.3, 'B': 0.4, 'C': 0.3},
                ('Low', 'Low', 'High'): {'A': 0.1, 'B': 0.3, 'C': 0.6},
                ('Low', 'Low', 'Low'): {'A': 0.2, 'B': 0.3, 'C': 0.5}
            }
        }
    
    def get_prob(self, node, parent_values, value):
        """
        Obtiene la probabilidad condicional P(node=value | parents=parent_values)
        """
        if not self.variables[node]:  # Si el nodo no tiene padres (es raíz)
            return self.cpts[node][value]
        
        # Si tiene padres, convertir los valores de los padres en una clave para la CPT
        if len(parent_values) == 1:
            key = parent_values[0]  # Si hay un solo padre
        else:
            key = tuple(parent_values)  # Si hay múltiples padres
            
        return self.cpts[node][key][value]
    
    def joint_probability(self, assignment):
        """
        Calcula la probabilidad conjunta de una asignación completa de valores
        usando la regla de la cadena.
        """
        prob = 1.0
        for node in self.variables:  # Iterar sobre cada variable
            parents = self.variables[node]  # Obtener los padres de la variable
            parent_values = [assignment[p] for p in parents]  # Valores de los padres
            node_value = assignment[node]  # Valor de la variable actual
            prob *= self.get_prob(node, parent_values, node_value)  # Multiplicar las probabilidades
        return prob
    
    def scenario_analysis(self, scenario):
        """
        Analiza un escenario particular y calcula su probabilidad conjunta.
        """
        print("\n=== Análisis de Escenario ===")
        print("Asignación de variables:")
        for var, val in scenario.items():  # Mostrar los valores asignados al escenario
            print(f"{var}: {val}")
        
        joint_prob = self.joint_probability(scenario)  # Calcular la probabilidad conjunta
        print(f"\nProbabilidad conjunta: {joint_prob:.6f}")
        
        # Mostrar la descomposición de la probabilidad conjunta usando la regla de la cadena
        print("\nDescomposición por regla de la cadena:")
        for node in self.variables:
            parents = self.variables[node]
            parent_values = [scenario[p] for p in parents] if parents else []
            node_value = scenario[node]
            prob = self.get_prob(node, parent_values, node_value)
            
            if parents:  # Si tiene padres, mostrar la probabilidad condicional
                parents_str = ", ".join([f"{p}={scenario[p]}" for p in parents])
                print(f"P({node}={node_value} | {parents_str}) = {prob:.4f}")
            else:  # Si no tiene padres, mostrar la probabilidad marginal
                print(f"P({node}={node_value}) = {prob:.4f}")
        
        return joint_prob

# Ejemplo de uso del modelo
if __name__ == "__main__":
    model = ChainRuleModel()
    
    # Escenario 1: Estudiante con buen rendimiento
    scenario_1 = {
        'Background': 'High',
        'Preparation': 'High',
        'Difficulty': 'Low',
        'Effort': 'High',
        'Skill': 'High',
        'Performance': 'A'
    }
    prob_1 = model.scenario_analysis(scenario_1)  # Analizar el escenario 1
    
    # Escenario 2: Estudiante con dificultades
    scenario_2 = {
        'Background': 'Low',
        'Preparation': 'Low',
        'Difficulty': 'High',
        'Effort': 'Low',
        'Skill': 'Low',
        'Performance': 'C'
    }
    prob_2 = model.scenario_analysis(scenario_2)  # Analizar el escenario 2