import numpy as np
from collections import defaultdict
from itertools import product

class AdvancedBayesianNetwork:
    def __init__(self):
        # Estructura de la red bayesiana: {nodo: [padres]}
        self.graph = {
            "Enfermedad1": [],  # Nodo raíz (sin padres)
            "Enfermedad2": [],  # Nodo raíz (sin padres)
            "Sintoma1": ["Enfermedad1", "Enfermedad2"],  # Nodo con dos padres
            "Sintoma2": ["Enfermedad1"],  # Nodo con un padre
            "Sintoma3": ["Enfermedad2"]   # Nodo con un padre
        }
        
        # Inicializar las Tablas de Probabilidad Condicional (CPTs)
        self.cpt = self._initialize_cpts()
        
        # Datos para aprendizaje de parámetros
        self.data = []
        self.learned = False

    def _initialize_cpts(self) -> dict:
        """Inicializa las Tablas de Probabilidad Condicional (CPTs) con valores predeterminados."""
        cpt = {
            "Enfermedad1": {"Prob": 0.02},  # Probabilidad marginal de Enfermedad1
            "Enfermedad2": {"Prob": 0.015},  # Probabilidad marginal de Enfermedad2
            "Sintoma1": {  # Probabilidades condicionales de Sintoma1 dado sus padres
                (True, True): 0.95,
                (True, False): 0.85,
                (False, True): 0.75,
                (False, False): 0.05
            },
            "Sintoma2": {  # Probabilidades condicionales de Sintoma2 dado su padre
                (True,): 0.8,
                (False,): 0.1
            },
            "Sintoma3": {  # Probabilidades condicionales de Sintoma3 dado su padre
                (True,): 0.7,
                (False,): 0.05
            }
        }
        return cpt

    def learn_from_data(self, data: list):
        """Aprende las CPTs a partir de datos observados utilizando máxima verosimilitud."""
        counts = defaultdict(lambda: defaultdict(int))
        
        # Contar ocurrencias de cada combinación de valores en los datos
        for sample in data:
            for node in self.graph:
                parents = self.graph[node]
                parent_values = tuple(sample[parent] for parent in parents) if parents else None
                counts[node][(parent_values, sample[node])] += 1
        
        # Calcular probabilidades condicionales a partir de las ocurrencias
        for node in self.graph:
            parents = self.graph[node]
            
            if not parents:  # Si el nodo no tiene padres (nodo raíz)
                total = counts[node][(None, True)] + counts[node][(None, False)]
                if total > 0:
                    self.cpt[node]["Prob"] = counts[node][(None, True)] / total
                continue
                
            # Para nodos con padres, calcular probabilidades condicionales
            unique_parent_values = set(
                key[0] for key in counts[node] 
                if key[0] is not None
            )
            
            for parent_values in unique_parent_values:
                total = counts[node][(parent_values, True)] + counts[node][(parent_values, False)]
                
                if total > 0:
                    self.cpt[node][parent_values] = {
                        True: counts[node][(parent_values, True)] / total,
                        False: counts[node][(parent_values, False)] / total
                    }
                else:
                    # Asignar valores por defecto si no hay datos suficientes
                    self.cpt[node][parent_values] = {True: 0.5, False: 0.5}
        
        self.learned = True

    def infer_exact(self, evidence: dict, target: str) -> float:
        """Realiza inferencia exacta utilizando enumeración de variables ocultas."""
        hidden_vars = [var for var in self.graph if var not in evidence]
        posterior = 0.0
        total_prob = 0.0
        
        # Generar todas las combinaciones posibles de valores para las variables ocultas
        for combo in product([False, True], repeat=len(hidden_vars)):
            scenario = dict(zip(hidden_vars, combo))
            scenario.update(evidence)
            
            # Calcular la probabilidad conjunta del escenario
            joint_prob = 1.0
            for node in self.graph:
                parents = self.graph[node]
                
                if not parents:  # Nodo raíz
                    prob = self.cpt[node]["Prob"]
                    joint_prob *= prob if scenario[node] else (1 - prob)
                else:
                    parent_values = tuple(scenario[parent] for parent in parents)
                    node_prob = self.cpt[node][parent_values][scenario[node]]
                    joint_prob *= node_prob if scenario[node] else (1 - node_prob)
            
            if scenario.get(target, False):  # Si el objetivo es verdadero en este escenario
                posterior += joint_prob
            
            total_prob += joint_prob
        
        # Retornar la probabilidad posterior normalizada
        return posterior / total_prob if total_prob > 0 else 0.0

    def gibbs_sampling(self, evidence: dict, target: str, iterations: int = 10000) -> float:
        """Realiza inferencia aproximada utilizando muestreo de Gibbs."""
        hidden_vars = [var for var in self.graph if var not in evidence]
        current_state = {var: np.random.rand() > 0.5 for var in hidden_vars}  # Estado inicial aleatorio
        current_state.update(evidence)
        
        counts = {True: 0, False: 0}
        
        for _ in range(iterations):
            for var in hidden_vars:
                # Calcular P(var | Markov Blanket)
                prob_true = self._gibbs_probability(var, True, current_state)
                prob_false = self._gibbs_probability(var, False, current_state)
                
                # Evitar división por cero
                total = prob_true + prob_false
                if total <= 0:
                    prob = 0.5  # Probabilidad uniforme si no hay información
                else:
                    prob = prob_true / total
                    
                # Actualizar el estado actual
                current_state[var] = np.random.rand() < prob
            
            # Contar ocurrencias del valor objetivo
            counts[current_state[target]] += 1
        
        # Retornar la proporción de iteraciones en las que el objetivo fue verdadero
        return counts[True] / iterations if iterations > 0 else 0.0

    def _gibbs_probability(self, node: str, value: bool, state: dict) -> float:
        """Calcula P(node=value | Markov Blanket)."""
        parents = self.graph[node]
        
        try:
            if not parents:  # Nodo raíz
                prob = self.cpt[node]["Prob"] if value else (1 - self.cpt[node]["Prob"])
            else:
                parent_values = tuple(state[parent] for parent in parents)
                prob = self.cpt[node][parent_values][value] if value else (1 - self.cpt[node][parent_values][value])
            
            # Multiplicar por las probabilidades de los hijos
            children = [n for n in self.graph if node in self.graph[n]]
            for child in children:
                child_parents = self.graph[child]
                child_parent_values = tuple(state[parent] for parent in child_parents)
                child_prob = self.cpt[child][child_parent_values][state[child]]
                prob *= child_prob if state[child] else (1 - child_prob)
            
            return max(prob, 0.0)  # Asegurar que la probabilidad no sea negativa
        except KeyError:
            # Si falta algún dato, devolver probabilidad neutra
            return 0.5

# --- Ejemplo de Uso Completo ---
if __name__ == "__main__":
    bn = AdvancedBayesianNetwork()
    
    # Datos sintéticos para aprendizaje
    synthetic_data = [
        {"Enfermedad1": False, "Enfermedad2": False, "Sintoma1": False, "Sintoma2": False, "Sintoma3": False},
        {"Enfermedad1": True, "Enfermedad2": False, "Sintoma1": True, "Sintoma2": True, "Sintoma3": False},
        {"Enfermedad1": False, "Enfermedad2": True, "Sintoma1": True, "Sintoma2": False, "Sintoma3": True},
        {"Enfermedad1": True, "Enfermedad2": True, "Sintoma1": True, "Sintoma2": True, "Sintoma3": True},
    ]
    
    print("=== Aprendiendo de los datos ===")
    bn.learn_from_data(synthetic_data)
    
    # Mostrar CPTs aprendidas
    print("\nCPTs aprendidas:")
    for node, probs in bn.cpt.items():
        print(f"{node}: {probs}")
    
    # Evidencia para inferencia
    evidence = {"Sintoma1": True, "Sintoma2": True, "Sintoma3": False}
    
    # Inferencia exacta
    print("\n=== Inferencia Exacta ===")
    prob_exact = bn.infer_exact(evidence, "Enfermedad1")
    print(f"P(Enfermedad1 | Evidencia): {prob_exact*100:.2f}%")
    
    # Inferencia aproximada
    print("\n=== Muestreo de Gibbs (10,000 iteraciones) ===")
    prob_gibbs = bn.gibbs_sampling(evidence, "Enfermedad1", iterations=10000)
    print(f"P(Enfermedad1 | Evidencia): {prob_gibbs*100:.2f}%")