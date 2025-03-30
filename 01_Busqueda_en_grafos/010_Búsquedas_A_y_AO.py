import heapq
from math import sqrt
from collections import defaultdict

class BusquedaAvanzada:
    def __init__(self, grafo=None, coordenadas=None, grafo_and_or=None, heuristicas=None):
        """
        Inicializa el buscador con:
        - grafo: Para A* (diccionario de adyacencia).
        - coordenadas: Para heurísticas geométricas (distancias entre nodos).
        - grafo_and_or: Para AO* (nodos AND/OR).
        - heuristicas: Valores heurísticos para AO*.
        """
        self.grafo = grafo
        self.coordenadas = coordenadas if coordenadas else {}
        self.grafo_and_or = grafo_and_or if grafo_and_or else {}
        self.heuristicas = heuristicas if heuristicas else {}

    # ------------------------------------------
    # ALGORITMO A*
    # ------------------------------------------
    def heuristica_euclidiana(self, a, b):
        """
        Calcula la distancia rectilínea (euclidiana) entre dos nodos.
        Args:
            a: Nodo actual.
            b: Nodo objetivo.
        Returns:
            Distancia euclidiana entre los nodos.
        """
        if a not in self.coordenadas or b not in self.coordenadas:
            return 0  # Si no hay coordenadas, retornar 0.
        x1, y1 = self.coordenadas[a]
        x2, y2 = self.coordenadas[b]
        return sqrt((x1 - x2)**2 + (y1 - y2)**2)

    def a_star(self, inicio, objetivo):
        """
        Implementación del algoritmo A* para grafos OR regulares.
        Combina el costo acumulado (g) y la heurística (h) para encontrar el camino más corto.
        Args:
            inicio: Nodo inicial.
            objetivo: Nodo objetivo.
        Returns:
            tuple: (camino, costo_total, nodos_expandidos).
        """
        # Cola de prioridad para explorar los nodos con menor costo f = g + h.
        cola = [(0, inicio, 0, [inicio])]  # (f, nodo, g, camino).
        visitados = {}  # Diccionario para registrar los costos acumulados de los nodos visitados.
        expandidos = 0  # Contador de nodos expandidos.

        while cola:
            # Extraer el nodo con el menor costo f de la cola de prioridad.
            f, actual, g, camino = heapq.heappop(cola)
            expandidos += 1  # Incrementar el contador de nodos expandidos.

            # Si llegamos al nodo objetivo, retornar el camino, el costo total y los nodos expandidos.
            if actual == objetivo:
                return camino, g, expandidos

            # Si el nodo ya fue visitado con un menor costo acumulado, lo ignoramos.
            if actual in visitados and visitados[actual] <= g:
                continue

            # Registrar el costo acumulado del nodo actual.
            visitados[actual] = g

            # Explorar los vecinos del nodo actual.
            for vecino in self.grafo[actual]:
                costo_arista = 1  # Asumimos costo uniforme por simplicidad.
                g_nuevo = g + costo_arista  # Calcular el nuevo costo acumulado g.
                h_nuevo = self.heuristica_euclidiana(vecino, objetivo)  # Calcular la heurística h.
                f_nuevo = g_nuevo + h_nuevo  # Calcular el costo total f = g + h.

                # Si el vecino no ha sido visitado o encontramos un menor costo acumulado, lo añadimos a la cola.
                if vecino not in visitados or g_nuevo < visitados.get(vecino, float('inf')):
                    heapq.heappush(cola, (f_nuevo, vecino, g_nuevo, camino + [vecino]))

        # Si no se encuentra un camino al objetivo, retornar None, costo infinito y nodos expandidos.
        return None, float('inf'), expandidos

    # ------------------------------------------
    # ALGORITMO AO*
    # ------------------------------------------
    def ao_star(self, inicio, objetivo):
        """
        Implementación del algoritmo AO* para grafos AND-OR.
        Args:
            inicio: Nodo inicial.
            objetivo: Nodo objetivo.
        Returns:
            dict: {'costo': X, 'camino': [...]}.
        """
        solucion = {}  # Diccionario para almacenar soluciones parciales.
        return self.__ao_star_rec(inicio, objetivo, solucion)

    def __ao_star_rec(self, nodo, objetivo, solucion):
        """
        Implementación recursiva del algoritmo AO*.
        Args:
            nodo: Nodo actual.
            objetivo: Nodo objetivo.
            solucion: Diccionario para almacenar soluciones parciales.
        Returns:
            dict: {'costo': X, 'camino': [...]}.
        """
        # Caso base: Si el nodo es el objetivo, retornar costo 0 y el camino.
        if nodo == objetivo:
            return {'costo': 0, 'camino': [nodo]}
        
        # Si ya se calculó la solución para este nodo, retornarla.
        if nodo in solucion:
            return solucion[nodo]

        # Inicializar con valores por defecto.
        mejor = {
            'costo': float('inf'),
            'camino': None,
            'explicacion': []
        }

        # Expandir según el tipo de nodo (AND/OR).
        for tipo, hijos in self.grafo_and_or[nodo]:
            if tipo == 'OR':
                # Para nodos OR, elegir el hijo con menor costo.
                for hijo in hijos:
                    resultado = self.__ao_star_rec(hijo, objetivo, solucion)
                    costo_total = resultado['costo'] + 1  # +1 por el costo de la arista.
                    
                    if costo_total < mejor['costo']:
                        mejor = {
                            'costo': costo_total,
                            'camino': [nodo] + resultado['camino'],
                            'explicacion': [f"OR: Elegido {hijo}"]
                        }
            else:  # AND
                # Para nodos AND, sumar los costos de todos los hijos.
                costos_hijos = []
                caminos_hijos = []
                explicaciones = []
                for hijo in hijos:
                    resultado = self.__ao_star_rec(hijo, objetivo, solucion)
                    costos_hijos.append(resultado['costo'])
                    caminos_hijos.append(resultado['camino'])
                    explicaciones.append(f"AND: Requerido {hijo} (costo {resultado['costo']})")
                
                costo_total = sum(costos_hijos) + len(hijos)  # Suma de costos + aristas.
                
                if costo_total < mejor['costo']:
                    mejor = {
                        'costo': costo_total,
                        'camino': [nodo] + [item for sublist in caminos_hijos for item in sublist],
                        'explicacion': explicaciones
                    }

        # Usar heurística si no hay hijos.
        if not self.grafo_and_or[nodo] and nodo in self.heuristicas:
            mejor = {
                'costo': self.heuristicas[nodo],
                'camino': [nodo],
                'explicacion': [f"Heurística: {self.heuristicas[nodo]}"]
            }

        # Guardar la solución para el nodo actual.
        solucion[nodo] = mejor
        return mejor

    def ejemplo_a_star():
        """
        Ejemplo de uso del algoritmo A*.
        """
        grafo = {
            'A': ['B', 'C'],
            'B': ['D', 'E'],
            'C': ['F'],
            'D': [],
            'E': ['F'],
            'F': []
        }

        coordenadas = {
            'A': (0, 0), 'B': (1, 0), 'C': (0, 1),
            'D': (2, 0), 'E': (1, 1), 'F': (2, 2)
        }

        buscador = BusquedaAvanzada(grafo=grafo, coordenadas=coordenadas)
        return buscador.a_star('A', 'F')

    def ejemplo_ao_star():
        """
        Ejemplo de uso del algoritmo AO*.
        """
        grafo_and_or = {
            'A': [('OR', ['B']), ('AND', ['C', 'D'])],
            'B': [('OR', ['E'])],
            'C': [('OR', ['F'])],
            'D': [('OR', ['F'])],
            'E': [],
            'F': []
        }

        heuristicas = {
            'A': 3, 'B': 2, 'C': 1,
            'D': 1, 'E': 0, 'F': 0
        }

        buscador = BusquedaAvanzada(grafo_and_or=grafo_and_or, heuristicas=heuristicas)
        return buscador.ao_star('A', 'F')

if __name__ == "__main__":
    print("=== EJECUCIÓN DE A* ===")
    camino_a, costo_a, expandidos_a = BusquedaAvanzada.ejemplo_a_star()
    print(f"Camino: {' → '.join(camino_a)}")
    print(f"Costo: {costo_a}")
    print(f"Nodos expandidos: {expandidos_a}")

    print("\n=== EJECUCIÓN DE AO* ===")
    resultado_ao = BusquedaAvanzada.ejemplo_ao_star()
    print(f"Camino: {' → '.join(resultado_ao['camino'])}")
    print(f"Costo total: {resultado_ao['costo']}")
    print("Explicación:")
    for paso in resultado_ao['explicacion']:
        print(f"- {paso}")