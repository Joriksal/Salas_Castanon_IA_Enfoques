# Importamos la librería numpy para realizar operaciones matemáticas y trabajar con números aleatorios.
import numpy as np

# Importamos defaultdict de collections para inicializar diccionarios con valores predeterminados.
from collections import defaultdict

# Importamos product de itertools para generar todas las combinaciones posibles de valores.
from itertools import product

class AdvancedBayesianNetwork:
    """
    Clase que representa una Red Bayesiana Avanzada.
    Permite realizar aprendizaje de parámetros, inferencia exacta y aproximada.
    """

    def __init__(self):
        """
        Constructor de la clase. Inicializa la estructura de la red bayesiana,
        las tablas de probabilidad condicional (CPTs) y los datos de aprendizaje.
        """
        # Estructura de la red bayesiana: cada nodo tiene una lista de sus padres.
        self.graph = {
            "Enfermedad1": [],  # Nodo raíz (sin padres)
            "Enfermedad2": [],  # Nodo raíz (sin padres)
            "Sintoma1": ["Enfermedad1", "Enfermedad2"],  # Nodo con dos padres
            "Sintoma2": ["Enfermedad1"],  # Nodo con un padre
            "Sintoma3": ["Enfermedad2"]   # Nodo con un padre
        }
        
        # Inicializamos las Tablas de Probabilidad Condicional (CPTs) con valores predeterminados.
        self.cpt = self._initialize_cpts()
        
        # Lista para almacenar datos observados para el aprendizaje de parámetros.
        self.data = []
        
        # Bandera para indicar si los parámetros han sido aprendidos.
        self.learned = False

    def _initialize_cpts(self) -> dict:
        """
        Inicializa las Tablas de Probabilidad Condicional (CPTs) con valores predeterminados.
        Retorna un diccionario que contiene las probabilidades marginales y condicionales.
        """
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
        """
        Aprende las Tablas de Probabilidad Condicional (CPTs) a partir de datos observados
        utilizando el método de máxima verosimilitud.
        
        Args:
            data (list): Lista de diccionarios donde cada diccionario representa un ejemplo observado.
        """
        # Diccionario para contar ocurrencias de combinaciones de valores.
        counts = defaultdict(lambda: defaultdict(int))
        
        # Iteramos sobre cada muestra en los datos.
        for sample in data:
            for node in self.graph:
                # Obtenemos los padres del nodo actual.
                parents = self.graph[node]
                
                # Obtenemos los valores de los padres en la muestra actual.
                parent_values = tuple(sample[parent] for parent in parents) if parents else None
                
                # Incrementamos el contador para la combinación actual de valores.
                counts[node][(parent_values, sample[node])] += 1
        
        # Calculamos las probabilidades condicionales a partir de las ocurrencias.
        for node in self.graph:
            parents = self.graph[node]
            
            if not parents:  # Si el nodo no tiene padres (nodo raíz)
                total = counts[node][(None, True)] + counts[node][(None, False)]
                if total > 0:
                    # Calculamos la probabilidad marginal.
                    self.cpt[node]["Prob"] = counts[node][(None, True)] / total
                continue
                
            # Para nodos con padres, calculamos probabilidades condicionales.
            unique_parent_values = set(
                key[0] for key in counts[node] 
                if key[0] is not None
            )
            
            for parent_values in unique_parent_values:
                total = counts[node][(parent_values, True)] + counts[node][(parent_values, False)]
                
                if total > 0:
                    # Calculamos las probabilidades condicionales.
                    self.cpt[node][parent_values] = {
                        True: counts[node][(parent_values, True)] / total,
                        False: counts[node][(parent_values, False)] / total
                    }
                else:
                    # Asignamos valores por defecto si no hay datos suficientes.
                    self.cpt[node][parent_values] = {True: 0.5, False: 0.5}
        
        # Marcamos que los parámetros han sido aprendidos.
        self.learned = True

    def infer_exact(self, evidence: dict, target: str) -> float:
        """
        Realiza inferencia exacta utilizando enumeración de variables ocultas.
        
        Args:
            evidence (dict): Diccionario con las variables observadas y sus valores.
            target (str): Nodo objetivo para el cual se desea calcular la probabilidad.
        
        Returns:
            float: Probabilidad posterior de la variable objetivo dado la evidencia.
        """
        # Identificamos las variables ocultas (no observadas).
        hidden_vars = [var for var in self.graph if var not in evidence]
        
        # Inicializamos las probabilidades acumuladas.
        posterior = 0.0
        total_prob = 0.0
        
        # Generamos todas las combinaciones posibles de valores para las variables ocultas.
        for combo in product([False, True], repeat=len(hidden_vars)):
            # Creamos un escenario combinando las variables ocultas y la evidencia.
            scenario = dict(zip(hidden_vars, combo))
            scenario.update(evidence)
            
            # Calculamos la probabilidad conjunta del escenario.
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
            
            # Si el objetivo es verdadero en este escenario, lo sumamos al posterior.
            if scenario.get(target, False):
                posterior += joint_prob
            
            # Sumamos la probabilidad conjunta al total.
            total_prob += joint_prob
        
        # Retornamos la probabilidad posterior normalizada.
        return posterior / total_prob if total_prob > 0 else 0.0

    def gibbs_sampling(self, evidence: dict, target: str, iterations: int = 10000) -> float:
        """
        Realiza inferencia aproximada utilizando muestreo de Gibbs.
        
        Args:
            evidence (dict): Diccionario con las variables observadas y sus valores.
            target (str): Nodo objetivo para el cual se desea calcular la probabilidad.
            iterations (int): Número de iteraciones del muestreo.
        
        Returns:
            float: Probabilidad aproximada de la variable objetivo dado la evidencia.
        """
        # Identificamos las variables ocultas (no observadas).
        hidden_vars = [var for var in self.graph if var not in evidence]
        
        # Inicializamos el estado actual con valores aleatorios para las variables ocultas.
        current_state = {var: np.random.rand() > 0.5 for var in hidden_vars}
        current_state.update(evidence)
        
        # Inicializamos el contador de ocurrencias del valor objetivo.
        counts = {True: 0, False: 0}
        
        # Iteramos el número especificado de veces.
        for _ in range(iterations):
            for var in hidden_vars:
                # Calculamos P(var | Markov Blanket).
                prob_true = self._gibbs_probability(var, True, current_state)
                prob_false = self._gibbs_probability(var, False, current_state)
                
                # Evitamos división por cero.
                total = prob_true + prob_false
                if total <= 0:
                    prob = 0.5  # Probabilidad uniforme si no hay información.
                else:
                    prob = prob_true / total
                    
                # Actualizamos el estado actual.
                current_state[var] = np.random.rand() < prob
            
            # Contamos ocurrencias del valor objetivo.
            counts[current_state[target]] += 1
        
        # Retornamos la proporción de iteraciones en las que el objetivo fue verdadero.
        return counts[True] / iterations if iterations > 0 else 0.0

    def _gibbs_probability(self, node: str, value: bool, state: dict) -> float:
        """
        Calcula P(node=value | Markov Blanket).
        
        Args:
            node (str): Nodo para el cual se calcula la probabilidad.
            value (bool): Valor del nodo (True o False).
            state (dict): Estado actual de las variables.
        
        Returns:
            float: Probabilidad condicional del nodo dado su Markov Blanket.
        """
        parents = self.graph[node]
        
        try:
            if not parents:  # Nodo raíz
                prob = self.cpt[node]["Prob"] if value else (1 - self.cpt[node]["Prob"])
            else:
                parent_values = tuple(state[parent] for parent in parents)
                prob = self.cpt[node][parent_values][value] if value else (1 - self.cpt[node][parent_values][value])
            
            # Multiplicamos por las probabilidades de los hijos.
            children = [n for n in self.graph if node in self.graph[n]]
            for child in children:
                child_parents = self.graph[child]
                child_parent_values = tuple(state[parent] for parent in child_parents)
                child_prob = self.cpt[child][child_parent_values][state[child]]
                prob *= child_prob if state[child] else (1 - child_prob)
            
            # Retornamos la probabilidad asegurándonos de que no sea negativa.
            return max(prob, 0.0)
        except KeyError:
            # Si falta algún dato, devolvemos una probabilidad neutra.
            return 0.5

# --- Ejemplo de Uso Completo ---
if __name__ == "__main__":
    # Creamos una instancia de la red bayesiana.
    bn = AdvancedBayesianNetwork()
    
    # Datos sintéticos para aprendizaje.
    synthetic_data = [
        {"Enfermedad1": False, "Enfermedad2": False, "Sintoma1": False, "Sintoma2": False, "Sintoma3": False},
        {"Enfermedad1": True, "Enfermedad2": False, "Sintoma1": True, "Sintoma2": True, "Sintoma3": False},
        {"Enfermedad1": False, "Enfermedad2": True, "Sintoma1": True, "Sintoma2": False, "Sintoma3": True},
        {"Enfermedad1": True, "Enfermedad2": True, "Sintoma1": True, "Sintoma2": True, "Sintoma3": True},
    ]
    
    print("=== Aprendiendo de los datos ===")
    # Aprendemos las CPTs a partir de los datos.
    bn.learn_from_data(synthetic_data)
    
    # Mostramos las CPTs aprendidas.
    print("\nCPTs aprendidas:")
    for node, probs in bn.cpt.items():
        print(f"{node}: {probs}")
    
    # Evidencia para inferencia.
    evidence = {"Sintoma1": True, "Sintoma2": True, "Sintoma3": False}
    
    # Realizamos inferencia exacta.
    print("\n=== Inferencia Exacta ===")
    prob_exact = bn.infer_exact(evidence, "Enfermedad1")
    print(f"P(Enfermedad1 | Evidencia): {prob_exact*100:.2f}%")
    
    # Realizamos inferencia aproximada utilizando muestreo de Gibbs.
    print("\n=== Muestreo de Gibbs (10,000 iteraciones) ===")
    prob_gibbs = bn.gibbs_sampling(evidence, "Enfermedad1", iterations=10000)
    print(f"P(Enfermedad1 | Evidencia): {prob_gibbs*100:.2f}%")