import numpy as np
from collections import defaultdict

# Clase para representar una red bayesiana
class BayesianNetwork:
    def __init__(self):
        self.nodes = []  # Lista de nodos en la red
        self.parents = defaultdict(list)  # Diccionario que mapea nodos a sus padres
        self.cpts = {}  # Diccionario que mapea nodos a sus tablas de probabilidad condicional (CPTs)
    
    def add_node(self, node, parents, cpt):
        """
        Agrega un nodo a la red bayesiana.
        
        Args:
            node: Nombre del nodo
            parents: Lista de nodos padres
            cpt: Función que define la tabla de probabilidad condicional (CPT)
        """
        self.nodes.append(node)
        self.parents[node] = parents
        self.cpts[node] = cpt

def likelihood_weighting_sampling(network, evidence, n_samples=10000):
    """
    Muestreo con ponderación de verosimilitud para redes bayesianas.
    
    Args:
        network: Objeto BayesianNetwork
        evidence: Diccionario de variables observadas {variable: valor}
        n_samples: Número de muestras a generar
    
    Returns:
        Lista de tuplas (muestra, peso) donde muestra es un diccionario {variable: valor}
    """
    weighted_samples = []  # Lista para almacenar las muestras ponderadas
    
    for _ in range(n_samples):
        sample = {}  # Diccionario para almacenar la muestra generada
        weight = 1.0  # Peso inicial de la muestra
        
        # Generar la muestra en orden topológico (padres antes que hijos)
        for node in network.nodes:
            if node in evidence:
                # Si el nodo es evidencia, ajustar el peso
                parent_values = [sample[p] for p in network.parents[node]]  # Valores de los padres
                prob = network.cpts[node](evidence[node], parent_values)  # Probabilidad condicional
                weight *= prob  # Actualizar el peso
                sample[node] = evidence[node]  # Asignar el valor observado
            else:
                # Si el nodo no es evidencia, muestrear su valor
                parent_values = [sample[p] for p in network.parents[node]]  # Valores de los padres
                sample[node] = network.cpts[node](None, parent_values)  # Muestrear valor
        
        # Agregar la muestra y su peso a la lista
        weighted_samples.append((sample, weight))
    
    return weighted_samples

# --------------------------------------------
# EJEMPLO COMPLETO: DIAGNÓSTICO MÉDICO
# --------------------------------------------

def create_medical_network():
    """
    Crea una red bayesiana para un ejemplo de diagnóstico médico.
    
    Returns:
        Objeto BayesianNetwork
    """
    bn = BayesianNetwork()
    
    # Nodos y sus padres
    # Nodo F1 (Factor 1) sin padres
    bn.add_node('F1', [], lambda val, _: 1 if val == 1 else 0)  # P(F1=1) = 0.3
    # Nodo F2 (Factor 2) sin padres
    bn.add_node('F2', [], lambda val, _: 1 if val == 1 else 0)  # P(F2=1) = 0.4
    # Nodo F3 (Factor 3) sin padres
    bn.add_node('F3', [], lambda val, _: 1 if val == 1 else 0)  # P(F3=1) = 0.1
    
    # Nodo E (Enfermedad) depende de F1, F2 y F3
    def e_cpt(val, parents):
        f1, f2, f3 = parents
        if val == 1:  # P(E=1|F1,F2,F3)
            return 0.01 + 0.2*f1 + 0.3*f2 + 0.1*f3  # Probabilidad acumulativa
        return 1 - (0.01 + 0.2*f1 + 0.3*f2 + 0.1*f3)
    
    bn.add_node('E', ['F1', 'F2', 'F3'], e_cpt)
    
    # Nodo S1 (Síntoma 1) depende de E
    def s1_cpt(val, parents):
        e = parents[0]
        if val == 1:  # P(S1=1|E)
            return 0.9 if e == 1 else 0.3
        return 0.1 if e == 1 else 0.7
    
    bn.add_node('S1', ['E'], s1_cpt)
    
    # Nodo S2 (Síntoma 2) depende de E
    def s2_cpt(val, parents):
        e = parents[0]
        if val == 1:  # P(S2=1|E)
            return 0.8 if e == 1 else 0.5
        return 0.2 if e == 1 else 0.5
    
    bn.add_node('S2', ['E'], s2_cpt)
    
    return bn

def main():
    # Crear la red bayesiana
    medical_net = create_medical_network()
    
    # Definir evidencia (S1=1, S2=0)
    evidence = {'S1': 1, 'S2': 0}
    
    # Generar muestras ponderadas
    samples = likelihood_weighting_sampling(medical_net, evidence, n_samples=10000)
    
    # Calcular P(E=1|evidencia)
    total_weight = sum(w for _, w in samples)  # Suma de todos los pesos
    e1_weight = sum(w for s, w in samples if s['E'] == 1)  # Suma de pesos donde E=1
    p_e1 = e1_weight / total_weight  # Probabilidad condicional P(E=1|evidencia)
    
    # Calcular P(F1=1|evidencia), P(F2=1|evidencia), P(F3=1|evidencia)
    p_f1 = sum(w for s, w in samples if s['F1'] == 1) / total_weight
    p_f2 = sum(w for s, w in samples if s['F2'] == 1) / total_weight
    p_f3 = sum(w for s, w in samples if s['F3'] == 1) / total_weight
    
    # Resultados de inferencia bayesiana
    print("\nRESULTADOS DE INFERENCIA BAYESIANA")
    print("----------------------------------")
    print(f"P(E=1 | S1=1, S2=0) = {p_e1:.4f}")
    print(f"P(F1=1 | S1=1, S2=0) = {p_f1:.4f}")
    print(f"P(F2=1 | S1=1, S2=0) = {p_f2:.4f}")
    print(f"P(F3=1 | S1=1, S2=0) = {p_f3:.4f}")
    
    # Estadísticas de muestreo
    weights = [w for _, w in samples]  # Lista de pesos
    print("\nESTADÍSTICAS DE MUESTREO")
    print("-----------------------")
    print(f"Muestras generadas: {len(samples)}")
    print(f"Peso máximo: {max(weights):.2e}")
    print(f"Peso mínimo: {min(weights):.2e}")
    print(f"Peso promedio: {np.mean(weights):.2e}")

if __name__ == "__main__":
    main()