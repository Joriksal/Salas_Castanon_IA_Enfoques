# Importamos la librería numpy para realizar cálculos matemáticos y estadísticos.
# np.mean se utiliza más adelante para calcular el promedio de los pesos.
import numpy as np

# Importamos defaultdict de collections, que es un diccionario especializado.
# Nos permite inicializar valores por defecto para claves que aún no existen.
from collections import defaultdict

# Definimos una clase para representar una red bayesiana.
class BayesianNetwork:
    def __init__(self):
        """
        Constructor de la clase BayesianNetwork.
        Inicializa las estructuras de datos necesarias para almacenar los nodos,
        sus relaciones (padres) y las tablas de probabilidad condicional (CPTs).
        """
        self.nodes = []  # Lista que almacena los nombres de los nodos en la red.
        self.parents = defaultdict(list)  # Diccionario que mapea cada nodo a su lista de padres.
        self.cpts = {}  # Diccionario que mapea cada nodo a su función CPT (tabla de probabilidad condicional).
    
    def add_node(self, node, parents, cpt):
        """
        Método para agregar un nodo a la red bayesiana.

        Args:
            node: Nombre del nodo (cadena de texto).
            parents: Lista de nodos padres de este nodo.
            cpt: Función que define la tabla de probabilidad condicional (CPT) del nodo.
        """
        self.nodes.append(node)  # Agregamos el nodo a la lista de nodos.
        self.parents[node] = parents  # Asociamos los padres al nodo en el diccionario.
        self.cpts[node] = cpt  # Asociamos la función CPT al nodo.

# Función para realizar muestreo con ponderación de verosimilitud.
def likelihood_weighting_sampling(network, evidence, n_samples=10000):
    """
    Implementa el algoritmo de muestreo con ponderación de verosimilitud para redes bayesianas.

    Args:
        network: Objeto de la clase BayesianNetwork que representa la red bayesiana.
        evidence: Diccionario que contiene las variables observadas y sus valores {variable: valor}.
        n_samples: Número de muestras a generar (por defecto, 10,000).

    Returns:
        Lista de tuplas (muestra, peso), donde:
        - muestra es un diccionario {variable: valor} que representa una muestra generada.
        - peso es un número que representa el peso asociado a la muestra.
    """
    weighted_samples = []  # Lista para almacenar las muestras generadas junto con sus pesos.

    # Generamos n_samples muestras.
    for _ in range(n_samples):
        sample = {}  # Diccionario para almacenar los valores generados para cada nodo.
        weight = 1.0  # Inicializamos el peso de la muestra en 1.

        # Recorremos los nodos en orden topológico (padres antes que hijos).
        for node in network.nodes:
            if node in evidence:
                # Si el nodo es una variable de evidencia:
                # Obtenemos los valores de los padres de este nodo en la muestra actual.
                parent_values = [sample[p] for p in network.parents[node]]
                # Calculamos la probabilidad condicional del valor observado.
                prob = network.cpts[node](evidence[node], parent_values)
                # Actualizamos el peso multiplicándolo por la probabilidad condicional.
                weight *= prob
                # Asignamos el valor observado al nodo en la muestra.
                sample[node] = evidence[node]
            else:
                # Si el nodo no es una variable de evidencia:
                # Obtenemos los valores de los padres de este nodo en la muestra actual.
                parent_values = [sample[p] for p in network.parents[node]]
                # Generamos un valor para el nodo usando su función CPT.
                sample[node] = network.cpts[node](None, parent_values)
        
        # Agregamos la muestra y su peso a la lista de muestras ponderadas.
        weighted_samples.append((sample, weight))
    
    return weighted_samples  # Devolvemos la lista de muestras ponderadas.

# --------------------------------------------
# EJEMPLO COMPLETO: DIAGNÓSTICO MÉDICO
# --------------------------------------------

def create_medical_network():
    """
    Crea una red bayesiana para un ejemplo de diagnóstico médico.

    Returns:
        Objeto de la clase BayesianNetwork que representa la red médica.
    """
    bn = BayesianNetwork()  # Creamos una instancia de la red bayesiana.

    # Definimos los nodos y sus relaciones (padres e hijos).

    # Nodo F1 (Factor 1) sin padres.
    # La función lambda define la probabilidad condicional P(F1=1) = 1.
    bn.add_node('F1', [], lambda val, _: 1 if val == 1 else 0)

    # Nodo F2 (Factor 2) sin padres.
    # La función lambda define la probabilidad condicional P(F2=1) = 1.
    bn.add_node('F2', [], lambda val, _: 1 if val == 1 else 0)

    # Nodo F3 (Factor 3) sin padres.
    # La función lambda define la probabilidad condicional P(F3=1) = 1.
    bn.add_node('F3', [], lambda val, _: 1 if val == 1 else 0)

    # Nodo E (Enfermedad) depende de F1, F2 y F3.
    def e_cpt(val, parents):
        """
        Tabla de probabilidad condicional para el nodo E (Enfermedad).
        Calcula P(E=1|F1,F2,F3) o P(E=0|F1,F2,F3).

        Args:
            val: Valor del nodo E (1 o 0).
            parents: Lista de valores de los nodos padres [F1, F2, F3].

        Returns:
            Probabilidad condicional P(E=val|F1,F2,F3).
        """
        f1, f2, f3 = parents  # Asignamos los valores de los padres.
        if val == 1:  # Si E=1:
            return 0.01 + 0.2*f1 + 0.3*f2 + 0.1*f3  # Probabilidad acumulativa.
        return 1 - (0.01 + 0.2*f1 + 0.3*f2 + 0.1*f3)  # Complemento para E=0.
    
    bn.add_node('E', ['F1', 'F2', 'F3'], e_cpt)  # Agregamos el nodo E.

    # Nodo S1 (Síntoma 1) depende de E.
    def s1_cpt(val, parents):
        """
        Tabla de probabilidad condicional para el nodo S1 (Síntoma 1).
        Calcula P(S1=1|E) o P(S1=0|E).

        Args:
            val: Valor del nodo S1 (1 o 0).
            parents: Lista de valores de los nodos padres [E].

        Returns:
            Probabilidad condicional P(S1=val|E).
        """
        e = parents[0]  # Asignamos el valor del nodo padre E.
        if val == 1:  # Si S1=1:
            return 0.9 if e == 1 else 0.3  # Probabilidad condicional.
        return 0.1 if e == 1 else 0.7  # Complemento para S1=0.
    
    bn.add_node('S1', ['E'], s1_cpt)  # Agregamos el nodo S1.

    # Nodo S2 (Síntoma 2) depende de E.
    def s2_cpt(val, parents):
        """
        Tabla de probabilidad condicional para el nodo S2 (Síntoma 2).
        Calcula P(S2=1|E) o P(S2=0|E).

        Args:
            val: Valor del nodo S2 (1 o 0).
            parents: Lista de valores de los nodos padres [E].

        Returns:
            Probabilidad condicional P(S2=val|E).
        """
        e = parents[0]  # Asignamos el valor del nodo padre E.
        if val == 1:  # Si S2=1:
            return 0.8 if e == 1 else 0.5  # Probabilidad condicional.
        return 0.2 if e == 1 else 0.5  # Complemento para S2=0.
    
    bn.add_node('S2', ['E'], s2_cpt)  # Agregamos el nodo S2.

    return bn  # Devolvemos la red bayesiana creada.

def main():
    """
    Función principal que ejecuta el ejemplo de diagnóstico médico.
    """
    # Creamos la red bayesiana médica.
    medical_net = create_medical_network()
    
    # Definimos la evidencia observada: S1=1 (síntoma 1 presente), S2=0 (síntoma 2 ausente).
    evidence = {'S1': 1, 'S2': 0}
    
    # Generamos muestras ponderadas usando el algoritmo de muestreo con ponderación de verosimilitud.
    samples = likelihood_weighting_sampling(medical_net, evidence, n_samples=10000)
    
    # Calculamos la probabilidad condicional P(E=1|evidencia).
    total_weight = sum(w for _, w in samples)  # Suma de todos los pesos.
    e1_weight = sum(w for s, w in samples if s['E'] == 1)  # Suma de pesos donde E=1.
    p_e1 = e1_weight / total_weight  # Probabilidad condicional P(E=1|evidencia).
    
    # Calculamos las probabilidades condicionales para F1, F2 y F3.
    p_f1 = sum(w for s, w in samples if s['F1'] == 1) / total_weight
    p_f2 = sum(w for s, w in samples if s['F2'] == 1) / total_weight
    p_f3 = sum(w for s, w in samples if s['F3'] == 1) / total_weight
    
    # Mostramos los resultados de la inferencia bayesiana.
    print("\nRESULTADOS DE INFERENCIA BAYESIANA")
    print("----------------------------------")
    print(f"P(E=1 | S1=1, S2=0) = {p_e1:.4f}")
    print(f"P(F1=1 | S1=1, S2=0) = {p_f1:.4f}")
    print(f"P(F2=1 | S1=1, S2=0) = {p_f2:.4f}")
    print(f"P(F3=1 | S1=1, S2=0) = {p_f3:.4f}")
    
    # Calculamos estadísticas de muestreo.
    weights = [w for _, w in samples]  # Lista de pesos.
    print("\nESTADÍSTICAS DE MUESTREO")
    print("-----------------------")
    print(f"Muestras generadas: {len(samples)}")
    print(f"Peso máximo: {max(weights):.2e}")
    print(f"Peso mínimo: {min(weights):.2e}")
    print(f"Peso promedio: {np.mean(weights):.2e}")

# Punto de entrada del programa.
if __name__ == "__main__":
    main()  # Llamamos a la función principal.