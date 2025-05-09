import heapq # Importa la biblioteca heapq para implementar una cola de prioridad (min-heap).
from math import sqrt  # Importa la función sqrt para calcular raíces cuadradas (usada en heurística euclidiana).

class BusquedaInformada:
    def __init__(self, grafo, coordenadas=None):
        """
        Constructor de la clase.
        - Inicializa el grafo y las coordenadas de los nodos.
        - Si no se proporcionan coordenadas, se inicializa como un diccionario vacío.
        """
        self.grafo = grafo  # Diccionario que representa el grafo como listas de adyacencia.
        self.coordenadas = coordenadas if coordenadas else {}  # Diccionario con posiciones (x, y) de los nodos.

    def heuristica_euclidiana(self, actual, objetivo):
        """
        Calcula la distancia euclidiana entre dos nodos.
        - Fórmula: sqrt((x2 - x1)^2 + (y2 - y1)^2).
        - Se usa para estimar la distancia en línea recta entre dos puntos.
        """
        x1, y1 = self.coordenadas[actual]  # Coordenadas del nodo actual.
        x2, y2 = self.coordenadas[objetivo]  # Coordenadas del nodo objetivo.
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)  # Retorna la distancia euclidiana.

    def heuristica_manhattan(self, actual, objetivo):
        """
        Calcula la distancia Manhattan entre dos nodos.
        - Fórmula: |x2 - x1| + |y2 - y1|.
        - Se usa para estimar la distancia en un entorno con movimientos en cuadrícula.
        """
        x1, y1 = self.coordenadas[actual]  # Coordenadas del nodo actual.
        x2, y2 = self.coordenadas[objetivo]  # Coordenadas del nodo objetivo.
        return abs(x1 - x2) + abs(y1 - y2)  # Retorna la distancia Manhattan.

    def a_estrella(self, inicio, objetivo, costos, tipo_heuristica='euclidiana'):
        """
        Implementa el algoritmo A* (A estrella).
        - Combina el costo acumulado (g) y una heurística (h) para encontrar el camino más corto.
        - Usa una cola de prioridad para explorar los nodos con menor costo total f = g + h.
        
        Args:
            inicio: Nodo inicial del camino.
            objetivo: Nodo objetivo al que se quiere llegar.
            costos: Diccionario con los costos de las aristas entre nodos.
            tipo_heuristica: Tipo de heurística a usar ('euclidiana' o 'manhattan').
        
        Returns:
            tuple: (camino, costo_total, nodos_expandidos).
        """
        # Selección de la heurística según el tipo especificado.
        if tipo_heuristica == 'euclidiana':
            heuristica = self.heuristica_euclidiana  # Usa la heurística euclidiana.
        else:
            heuristica = self.heuristica_manhattan  # Usa la heurística Manhattan.

        # Cola de prioridad para explorar los nodos con menor costo f = g + h.
        cola_prioridad = []
        # Insertar el nodo inicial con costo f = 0, g = 0 y el camino inicial.
        heapq.heappush(cola_prioridad, (0, inicio, 0, [inicio]))  # (f, nodo, g, camino)
        # Diccionario para registrar los costos acumulados de los nodos visitados.
        visitados = {}
        # Contador de nodos expandidos.
        nodos_expandidos = 0

        while cola_prioridad:
            # Extraer el nodo con el menor costo f de la cola de prioridad.
            _, actual, g_actual, camino = heapq.heappop(cola_prioridad)
            nodos_expandidos += 1  # Incrementar el contador de nodos expandidos.

            # Si llegamos al nodo objetivo, retornamos el camino, el costo total y los nodos expandidos.
            if actual == objetivo:
                return camino, g_actual, nodos_expandidos

            # Si el nodo ya fue visitado con un menor costo acumulado, lo ignoramos.
            if actual in visitados and visitados[actual] < g_actual:
                continue

            # Registrar el costo acumulado del nodo actual.
            visitados[actual] = g_actual

            # Explorar los vecinos del nodo actual.
            for vecino in self.grafo[actual]:
                # Obtener el costo de la arista entre el nodo actual y el vecino.
                costo_arista = costos.get((actual, vecino), 1)  # Default 1 si no se especifica costo.
                # Calcular el nuevo costo acumulado g.
                g_nuevo = g_actual + costo_arista
                # Calcular el valor heurístico h para el vecino.
                h_nuevo = heuristica(vecino, objetivo)
                # Calcular el costo total f = g + h.
                f_nuevo = g_nuevo + h_nuevo

                # Si el vecino no ha sido visitado o encontramos un menor costo acumulado, lo añadimos a la cola.
                if vecino not in visitados or g_nuevo < visitados[vecino]:
                    heapq.heappush(cola_prioridad, (f_nuevo, vecino, g_nuevo, camino + [vecino]))

        # Si no se encuentra un camino al objetivo, retornar None, costo infinito y nodos expandidos.
        return None, float('inf'), nodos_expandidos

# Ejemplo de uso
if __name__ == "__main__":
    # Coordenadas de las ciudades (nodos) en un plano 2D.
    coordenadas_ciudades = {
        'A': (0, 0),  # Nodo A en (0, 0).
        'B': (2, 0),  # Nodo B en (2, 0).
        'C': (4, 3),  # Nodo C en (4, 3).
        'D': (2, 3),  # Nodo D en (2, 3).
        'E': (5, 0),  # Nodo E en (5, 0).
        'F': (6, 4)   # Nodo F en (6, 4).
    }

    # Grafo de ciudades representado como un diccionario de listas de adyacencia.
    grafo_ciudades = {
        'A': ['B', 'D'],  # Nodo A conectado a B y D.
        'B': ['A', 'C', 'E'],  # Nodo B conectado a A, C y E.
        'C': ['B', 'D', 'F'],  # Nodo C conectado a B, D y F.
        'D': ['A', 'C'],  # Nodo D conectado a A y C.
        'E': ['B', 'F'],  # Nodo E conectado a B y F.
        'F': ['C', 'E']   # Nodo F conectado a C y E.
    }

    # Costos entre las aristas del grafo.
    costos_ciudades = {
        ('A','B'): 2, ('A','D'): 3,
        ('B','A'): 2, ('B','C'): 5, ('B','E'): 3,
        ('C','B'): 5, ('C','D'): 1, ('C','F'): 4,
        ('D','A'): 3, ('D','C'): 1,
        ('E','B'): 3, ('E','F'): 2,
        ('F','C'): 4, ('F','E'): 2
    }

    # Crear una instancia de la clase BusquedaInformada.
    buscador = BusquedaInformada(grafo_ciudades, coordenadas_ciudades)
    inicio = 'A'  # Nodo inicial.
    objetivo = 'F'  # Nodo objetivo.

    # Ejecutar A* con heurística euclidiana.
    print("=== A* con Heurística Euclidiana ===")
    camino, costo, expandidos = buscador.a_estrella(inicio, objetivo, costos_ciudades)
    print(f"Camino: {' → '.join(camino)}")  # Imprime el camino encontrado.
    print(f"Costo total: {costo}")  # Imprime el costo total del camino.
    print(f"Nodos expandidos: {expandidos}")  # Imprime el número de nodos expandidos.

    # Ejecutar A* con heurística Manhattan.
    print("\n=== A* con Heurística Manhattan ===")
    camino, costo, expandidos = buscador.a_estrella(inicio, objetivo, costos_ciudades, 'manhattan')
    print(f"Camino: {' → '.join(camino)}")  # Imprime el camino encontrado.
    print(f"Costo total: {costo}")  # Imprime el costo total del camino.
    print(f"Nodos expandidos: {expandidos}")  # Imprime el número de nodos expandidos.