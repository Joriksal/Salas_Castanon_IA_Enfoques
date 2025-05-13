# Definición de una clase en Python. Las clases son plantillas para crear objetos.
class ChainRuleModel:
    # Método especial __init__: se ejecuta automáticamente al crear una instancia de la clase.
    # Sirve para inicializar los atributos del objeto.
    def __init__(self):
        # Diccionario que define las variables del modelo y sus dependencias (padres).
        # Las claves son los nombres de las variables, y los valores son listas de sus padres.
        self.variables = {
            'Background': [],  # 'Background' no depende de ninguna otra variable.
            'Preparation': ['Background'],  # 'Preparation' depende de 'Background'.
            'Difficulty': [],  # 'Difficulty' tampoco tiene padres.
            'Effort': ['Preparation', 'Difficulty'],  # 'Effort' depende de 'Preparation' y 'Difficulty'.
            'Skill': ['Effort', 'Background'],  # 'Skill' depende de 'Effort' y 'Background'.
            'Performance': ['Skill', 'Effort', 'Difficulty']  # 'Performance' depende de varias variables.
        }
        
        # Tablas de probabilidad condicional (CPTs, por sus siglas en inglés).
        # Estas tablas contienen las probabilidades de cada variable dado el valor de sus padres.
        self.cpts = {
            # Probabilidades marginales de 'Background' (sin padres).
            'Background': {'High': 0.3, 'Low': 0.7},
            
            # Probabilidades condicionales de 'Preparation' dado 'Background'.
            'Preparation': {
                'High': {'High': 0.8, 'Low': 0.2},  # Si 'Background' es 'High'.
                'Low': {'High': 0.4, 'Low': 0.6}   # Si 'Background' es 'Low'.
            },
            
            # Probabilidades marginales de 'Difficulty' (sin padres).
            'Difficulty': {'High': 0.4, 'Low': 0.6},
            
            # Probabilidades condicionales de 'Effort' dado 'Preparation' y 'Difficulty'.
            'Effort': {
                ('High', 'High'): {'High': 0.7, 'Low': 0.3},
                ('High', 'Low'): {'High': 0.9, 'Low': 0.1},
                ('Low', 'High'): {'High': 0.5, 'Low': 0.5},
                ('Low', 'Low'): {'High': 0.8, 'Low': 0.2}
            },
            
            # Probabilidades condicionales de 'Skill' dado 'Effort' y 'Background'.
            'Skill': {
                ('High', 'High'): {'High': 0.95, 'Medium': 0.05, 'Low': 0.0},
                ('High', 'Low'): {'High': 0.7, 'Medium': 0.25, 'Low': 0.05},
                ('Low', 'High'): {'High': 0.5, 'Medium': 0.3, 'Low': 0.2},
                ('Low', 'Low'): {'High': 0.2, 'Medium': 0.4, 'Low': 0.4}
            },
            
            # Probabilidades condicionales de 'Performance' dado 'Skill', 'Effort' y 'Difficulty'.
            'Performance': {
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
    
    # Método para obtener la probabilidad condicional de una variable dado el valor de sus padres.
    def get_prob(self, node, parent_values, value):
        """
        Obtiene la probabilidad condicional P(node=value | parents=parent_values).
        """
        # Si la variable no tiene padres, devolver la probabilidad marginal.
        if not self.variables[node]:
            return self.cpts[node][value]
        
        # Si tiene padres, convertir los valores de los padres en una clave para la CPT.
        if len(parent_values) == 1:
            key = parent_values[0]  # Si hay un solo padre.
        else:
            key = tuple(parent_values)  # Si hay múltiples padres.
        
        # Devolver la probabilidad condicional desde la tabla CPT.
        return self.cpts[node][key][value]
    
    # Método para calcular la probabilidad conjunta de una asignación completa de valores.
    def joint_probability(self, assignment):
        """
        Calcula la probabilidad conjunta de una asignación completa de valores
        usando la regla de la cadena.
        """
        prob = 1.0  # Inicializar la probabilidad conjunta como 1 (neutro multiplicativo).
        for node in self.variables:  # Iterar sobre cada variable en el modelo.
            parents = self.variables[node]  # Obtener los padres de la variable.
            parent_values = [assignment[p] for p in parents]  # Valores de los padres.
            node_value = assignment[node]  # Valor de la variable actual.
            prob *= self.get_prob(node, parent_values, node_value)  # Multiplicar las probabilidades.
        return prob  # Devolver la probabilidad conjunta.
    
    # Método para analizar un escenario y calcular su probabilidad conjunta.
    def scenario_analysis(self, scenario):
        """
        Analiza un escenario particular y calcula su probabilidad conjunta.
        """
        print("\n=== Análisis de Escenario ===")
        print("Asignación de variables:")
        for var, val in scenario.items():  # Mostrar los valores asignados al escenario.
            print(f"{var}: {val}")
        
        # Calcular la probabilidad conjunta del escenario.
        joint_prob = self.joint_probability(scenario)
        print(f"\nProbabilidad conjunta: {joint_prob:.6f}")
        
        # Mostrar la descomposición de la probabilidad conjunta usando la regla de la cadena.
        print("\nDescomposición por regla de la cadena:")
        for node in self.variables:
            parents = self.variables[node]
            parent_values = [scenario[p] for p in parents] if parents else []
            node_value = scenario[node]
            prob = self.get_prob(node, parent_values, node_value)
            
            if parents:  # Si tiene padres, mostrar la probabilidad condicional.
                parents_str = ", ".join([f"{p}={scenario[p]}" for p in parents])
                print(f"P({node}={node_value} | {parents_str}) = {prob:.4f}")
            else:  # Si no tiene padres, mostrar la probabilidad marginal.
                print(f"P({node}={node_value}) = {prob:.4f}")
        
        return joint_prob  # Devolver la probabilidad conjunta.

# Bloque principal para ejecutar el código si se ejecuta directamente.
if __name__ == "__main__":
    # Crear una instancia del modelo.
    model = ChainRuleModel()
    
    # Escenario 1: Estudiante con buen rendimiento.
    scenario_1 = {
        'Background': 'High',
        'Preparation': 'High',
        'Difficulty': 'Low',
        'Effort': 'High',
        'Skill': 'High',
        'Performance': 'A'
    }
    # Analizar el escenario 1.
    prob_1 = model.scenario_analysis(scenario_1)
    
    # Escenario 2: Estudiante con dificultades.
    scenario_2 = {
        'Background': 'Low',
        'Preparation': 'Low',
        'Difficulty': 'High',
        'Effort': 'Low',
        'Skill': 'Low',
        'Performance': 'C'
    }
    # Analizar el escenario 2.
    prob_2 = model.scenario_analysis(scenario_2)