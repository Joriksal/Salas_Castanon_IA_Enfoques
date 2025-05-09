import heapq  # Librería para manejar colas de prioridad (min-heaps).
from math import sqrt  # Para calcular la raíz cuadrada en la heurística euclidiana.

class BusquedaVoraz:
    def __init__(self, grafo, coordenadas=None):
        """
        Constructor de la clase.
        - Inicializa el grafo y las coordenadas de los nodos.
        - Si no se proporcionan coordenadas, se inicializa como un diccionario vacío.
        """
        self.grafo = grafo  # Diccionario que representa el grafo como listas de adyacencia.
        self.coordenadas = coordenadas if coordenadas else {}  # Diccionario con coordenadas de los nodos.

    def heuristica_euclidiana(self, actual, objetivo):
        """
        Calcula la distancia euclidiana entre dos nodos.
        - Si no hay coordenadas para los nodos, retorna 0 (búsqueda no informada).
        - Usa la fórmula de distancia euclidiana: sqrt((x2 - x1)^2 + (y2 - y1)^2).
        """
        if actual not in self.coordenadas or objetivo not in self.coordenadas:
            return 0  # Retorna 0 si no hay coordenadas disponibles.
        x1, y1 = self.coordenadas[actual]  # Coordenadas del nodo actual.
        x2, y2 = self.coordenadas[objetivo]  # Coordenadas del nodo objetivo.
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)  # Calcula la distancia euclidiana.

    def buscar(self, inicio, objetivo, tipo_heuristica='euclidiana'):
        """
        Implementa el algoritmo de Búsqueda Voraz (Primero el Mejor).
        - Explora los nodos basándose únicamente en la heurística h(n).
        - Usa una cola de prioridad para seleccionar el nodo con menor h(n).

        Args:
            inicio: Nodo inicial desde donde comienza la búsqueda.
            objetivo: Nodo objetivo al que se desea llegar.
            tipo_heuristica: Tipo de heurística a usar ('euclidiana' o 'manhattan').

        Returns:
            tuple: (camino, nodos_expandidos)
            - camino: Lista de nodos que forman el camino encontrado.
            - nodos_expandidos: Número de nodos que fueron explorados.
        """
        # Selecciona la función heurística según el tipo especificado.
        if tipo_heuristica == 'euclidiana':
            heuristica = self.heuristica_euclidiana  # Usa la heurística euclidiana.
        else:
            # Define la heurística Manhattan como una función lambda.
            heuristica = lambda a, b: abs(self.coordenadas[a][0] - self.coordenadas[b][0]) + \
                                      abs(self.coordenadas[a][1] - self.coordenadas[b][1])

        # Cola de prioridad para manejar los nodos según su valor heurístico h(n).
        cola_prioridad = []
        # Inserta el nodo inicial en la cola con h = 0 y el camino inicial.
        heapq.heappush(cola_prioridad, (0, inicio, [inicio]))  # (h, nodo, camino)
        visitados = set()  # Conjunto para registrar los nodos ya visitados.
        nodos_expandidos = 0  # Contador de nodos expandidos.

        while cola_prioridad:  # Mientras haya nodos en la cola de prioridad.
            # Extrae el nodo con el menor valor de h(n).
            _, actual, camino = heapq.heappop(cola_prioridad)
            nodos_expandidos += 1  # Incrementa el contador de nodos expandidos.

            # Si el nodo actual es el objetivo, retorna el camino y los nodos expandidos.
            if actual == objetivo:
                return camino, nodos_expandidos

            # Si el nodo no ha sido visitado, procesarlo.
            if actual not in visitados:
                visitados.add(actual)  # Marca el nodo como visitado.
                # Explora los vecinos del nodo actual.
                for vecino in self.grafo[actual]:
                    if vecino not in visitados:  # Solo procesa vecinos no visitados.
                        # Calcula el valor heurístico h(n) para el vecino.
                        h = heuristica(vecino, objetivo)
                        # Añade el vecino a la cola de prioridad con su camino actualizado.
                        heapq.heappush(cola_prioridad, (h, vecino, camino + [vecino]))

        # Si no se encuentra un camino al objetivo, retorna None y los nodos expandidos.
        return None, nodos_expandidos

# Ejemplo de uso
if __name__ == "__main__":
    # Coordenadas de los nodos (ciudades) en un plano 2D.
    coordenadas = {
        'A': (0, 0),  # Nodo A en (0, 0).
        'B': (2, 0),  # Nodo B en (2, 0).
        'C': (4, 3),  # Nodo C en (4, 3).
        'D': (2, 3),  # Nodo D en (2, 3).
        'E': (5, 0),  # Nodo E en (5, 0).
        'F': (6, 4)   # Nodo F en (6, 4).
    }

    # Grafo representado como un diccionario de listas de adyacencia.
    grafo = {
        'A': ['B', 'D'],  # Nodo A conectado a B y D.
        'B': ['A', 'C', 'E'],  # Nodo B conectado a A, C y E.
        'C': ['B', 'D', 'F'],  # Nodo C conectado a B, D y F.
        'D': ['A', 'C'],  # Nodo D conectado a A y C.
        'E': ['B', 'F'],  # Nodo E conectado a B y F.
        'F': ['C', 'E']   # Nodo F conectado a C y E.
    }

    # Crear una instancia de la clase BusquedaVoraz.
    buscador = BusquedaVoraz(grafo, coordenadas)
    inicio = 'A'  # Nodo inicial.
    objetivo = 'F'  # Nodo objetivo.

    print("=== Búsqueda Voraz (Primero el Mejor) ===")
    print(f"De {inicio} a {objetivo}\n")

    # Prueba con heurística euclidiana.
    camino, expandidos = buscador.buscar(inicio, objetivo, 'euclidiana')
    print(f"Con heurística euclidiana:")
    print(f"Camino: {' → '.join(camino) if camino else 'No encontrado'}")
    print(f"Nodos expandidos: {expandidos}")

    # Prueba con heurística Manhattan.
    camino, expandidos = buscador.buscar(inicio, objetivo, 'manhattan')
    print(f"\nCon heurística Manhattan:")
    print(f"Camino: {' → '.join(camino) if camino else 'No encontrado'}")
    print(f"Nodos expandidos: {expandidos}")