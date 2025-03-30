import heapq
from math import sqrt

class BusquedaVoraz:
    def __init__(self, grafo, coordenadas=None):
        """
        Inicializa el buscador con:
        - grafo: Diccionario de listas de adyacencia que representa el grafo.
        - coordenadas: Diccionario con posiciones (x, y) de los nodos para calcular heurísticas geométricas.
        """
        self.grafo = grafo
        self.coordenadas = coordenadas if coordenadas else {}

    def heuristica_euclidiana(self, actual, objetivo):
        """
        Calcula la distancia euclidiana (rectilínea) entre dos nodos.
        Args:
            actual: Nodo actual.
            objetivo: Nodo objetivo.
        Returns:
            Distancia euclidiana entre los nodos.
        """
        # Si no hay coordenadas para los nodos, retornar 0 (búsqueda no informada).
        if actual not in self.coordenadas or objetivo not in self.coordenadas:
            return 0
        # Obtener las coordenadas de los nodos.
        x1, y1 = self.coordenadas[actual]
        x2, y2 = self.coordenadas[objetivo]
        # Calcular la distancia euclidiana.
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def buscar(self, inicio, objetivo, tipo_heuristica='euclidiana'):
        """
        Algoritmo de Búsqueda Voraz (Primero el Mejor).
        Explora los nodos basándose únicamente en la heurística h(n).
        
        Args:
            inicio: Nodo inicial.
            objetivo: Nodo objetivo.
            tipo_heuristica: 'euclidiana' o 'manhattan'.
        
        Returns:
            tuple: (camino, nodos_expandidos)
        """
        # Configurar la heurística según el tipo especificado.
        if tipo_heuristica == 'euclidiana':
            heuristica = self.heuristica_euclidiana
        else:
            # Heurística Manhattan: suma de diferencias absolutas entre coordenadas.
            heuristica = lambda a, b: abs(self.coordenadas[a][0] - self.coordenadas[b][0]) + \
                                      abs(self.coordenadas[a][1] - self.coordenadas[b][1])

        # Cola de prioridad basada únicamente en la heurística h(n).
        cola_prioridad = []
        # Insertar el nodo inicial con h = 0 y el camino inicial.
        heapq.heappush(cola_prioridad, (0, inicio, [inicio]))  # (h, nodo, camino)
        # Conjunto para registrar los nodos visitados.
        visitados = set()
        # Contador de nodos expandidos.
        nodos_expandidos = 0

        while cola_prioridad:
            # Extraer el nodo con el menor valor de h de la cola de prioridad.
            _, actual, camino = heapq.heappop(cola_prioridad)
            nodos_expandidos += 1  # Incrementar el contador de nodos expandidos.

            # Si llegamos al nodo objetivo, retornar el camino y los nodos expandidos.
            if actual == objetivo:
                return camino, nodos_expandidos

            # Si el nodo no ha sido visitado, procesarlo.
            if actual not in visitados:
                visitados.add(actual)
                # Explorar los vecinos del nodo actual.
                for vecino in self.grafo[actual]:
                    if vecino not in visitados:
                        # Calcular el valor heurístico h para el vecino.
                        h = heuristica(vecino, objetivo)
                        # Añadir el vecino a la cola de prioridad.
                        heapq.heappush(cola_prioridad, (h, vecino, camino + [vecino]))

        # Si no se encuentra un camino al objetivo, retornar None y los nodos expandidos.
        return None, nodos_expandidos

# Ejemplo de uso
if __name__ == "__main__":
    # Coordenadas de los nodos (ciudades) en un plano 2D.
    coordenadas = {
        'A': (0, 0),
        'B': (2, 0),
        'C': (4, 3),
        'D': (2, 3),
        'E': (5, 0),
        'F': (6, 4)
    }

    # Grafo representado como un diccionario de listas de adyacencia.
    grafo = {
        'A': ['B', 'D'],
        'B': ['A', 'C', 'E'],
        'C': ['B', 'D', 'F'],
        'D': ['A', 'C'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }

    # Crear una instancia de la clase BusquedaVoraz.
    buscador = BusquedaVoraz(grafo, coordenadas)
    inicio = 'A'
    objetivo = 'F'

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