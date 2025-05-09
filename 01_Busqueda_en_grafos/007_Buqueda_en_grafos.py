from collections import deque  # Para manejar colas en BFS.
import heapq  # Para manejar colas de prioridad en UCS.

class BusquedaGrafo:
    def __init__(self, grafo):
        """
        Inicializa el buscador con un grafo.
        Args:
            grafo (dict): Diccionario de listas de adyacencia.
                         Ejemplo: {'A': ['B', 'C'], 'B': ['A', 'D']}
        """
        self.grafo = grafo  # Almacena el grafo como un diccionario de listas de adyacencia.
    
    def bfs(self, inicio, objetivo):
        """
        Búsqueda en Anchura (Breadth-First Search).
        Explora los nodos nivel por nivel, asegurando encontrar el camino más corto
        en términos de número de aristas en un grafo no ponderado.
        Args:
            inicio (str): Nodo inicial desde donde comienza la búsqueda.
            objetivo (str): Nodo objetivo que se desea encontrar.
        Returns:
            list: Camino más corto desde el nodo inicial al objetivo, o None si no se encuentra.
        """
        # Cola para almacenar los nodos a explorar, junto con el camino actual.
        cola = deque([(inicio, [inicio])])  # deque: estructura eficiente para operaciones FIFO.
        # Conjunto para registrar los nodos visitados.
        visitados = set()  # set: estructura para evitar procesar nodos repetidos.
        
        while cola:  # Mientras haya nodos en la cola.
            # Extraer el nodo actual y el camino desde la cola.
            nodo, camino = cola.popleft()  # popleft: elimina y retorna el primer elemento de la cola.
            
            # Si encontramos el nodo objetivo, retornamos el camino.
            if nodo == objetivo:
                return camino
            
            # Si el nodo no ha sido visitado, lo procesamos.
            if nodo not in visitados:
                visitados.add(nodo)  # Añadir el nodo al conjunto de visitados.
                # Añadir los vecinos no visitados a la cola.
                for vecino in self.grafo[nodo]:  # Iterar sobre los vecinos del nodo actual.
                    if vecino not in visitados:  # Solo procesar vecinos no visitados.
                        cola.append((vecino, camino + [vecino]))  # Agregar vecino y camino actualizado.
        # Si no se encuentra el objetivo, retornar None.
        return None
    
    def dfs(self, inicio, objetivo, limite=None):
        """
        Búsqueda en Profundidad (Depth-First Search).
        Explora los nodos en profundidad antes de retroceder.
        Args:
            inicio (str): Nodo inicial desde donde comienza la búsqueda.
            objetivo (str): Nodo objetivo que se desea encontrar.
            limite (int, opcional): Límite de profundidad para evitar ciclos infinitos.
        Returns:
            list: Camino desde el nodo inicial al objetivo, o None si no se encuentra.
        """
        # Pila para almacenar los nodos a explorar, junto con el camino actual.
        pila = [(inicio, [inicio])]  # Lista usada como pila (LIFO).
        # Conjunto para registrar los nodos visitados.
        visitados = set()
        
        while pila:  # Mientras haya nodos en la pila.
            # Extraer el nodo actual y el camino desde la pila.
            nodo, camino = pila.pop()  # pop: elimina y retorna el último elemento de la pila.
            
            # Si encontramos el nodo objetivo, retornamos el camino.
            if nodo == objetivo:
                return camino
                
            # Si el nodo no ha sido visitado, lo procesamos.
            if nodo not in visitados:
                visitados.add(nodo)  # Añadir el nodo al conjunto de visitados.
                # Si no se ha alcanzado el límite de profundidad, explorar vecinos.
                if limite is None or len(camino) < limite:  # Verificar si se respeta el límite.
                    for vecino in reversed(self.grafo[nodo]):  # reversed: explora vecinos en orden inverso.
                        if vecino not in visitados:
                            pila.append((vecino, camino + [vecino]))  # Agregar vecino y camino actualizado.
        # Si no se encuentra el objetivo, retornar None.
        return None
    
    def ucs(self, inicio, objetivo, costos):
        """
        Búsqueda de Costo Uniforme (Uniform Cost Search).
        Encuentra el camino de menor costo en un grafo ponderado.
        Args:
            inicio (str): Nodo inicial desde donde comienza la búsqueda.
            objetivo (str): Nodo objetivo que se desea encontrar.
            costos (dict): Diccionario de costos entre nodos.
                          Ejemplo: {('A','B'): 3, ('A','C'): 5}
        Returns:
            tuple: Camino más corto y su costo total, o (None, infinito) si no se encuentra.
        """
        # Cola de prioridad para explorar los nodos con menor costo acumulado.
        cola_prioridad = []  # Lista usada como cola de prioridad.
        # Insertar el nodo inicial con costo 0.
        heapq.heappush(cola_prioridad, (0, inicio, [inicio]))  # heappush: inserta en la cola de prioridad.
        # Conjunto para registrar los nodos visitados.
        visitados = set()
        
        while cola_prioridad:  # Mientras haya nodos en la cola de prioridad.
            # Extraer el nodo con el menor costo acumulado.
            costo, nodo, camino = heapq.heappop(cola_prioridad)  # heappop: extrae el elemento con menor prioridad.
            
            # Si encontramos el nodo objetivo, retornamos el camino y el costo.
            if nodo == objetivo:
                return camino, costo
                
            # Si el nodo no ha sido visitado, lo procesamos.
            if nodo not in visitados:
                visitados.add(nodo)  # Añadir el nodo al conjunto de visitados.
                # Explorar los vecinos del nodo actual.
                for vecino in self.grafo[nodo]:
                    if vecino not in visitados:
                        # Calcular el nuevo costo acumulado.
                        nuevo_costo = costo + costos.get((nodo, vecino), 1)  # Default 1 si no se especifica costo.
                        # Añadir el vecino a la cola de prioridad.
                        heapq.heappush(cola_prioridad, (nuevo_costo, vecino, camino + [vecino]))
        # Si no se encuentra el objetivo, retornar None y un costo infinito.
        return None, float('inf')

# Ejemplo de uso
if __name__ == "__main__":
    # Grafo no ponderado representado como un diccionario de listas de adyacencia.
    grafo_no_ponderado = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    # Costos para UCS representados como un diccionario de tuplas.
    costos = {
        ('A','B'): 1, ('A','C'): 4,
        ('B','A'): 1, ('B','D'): 2, ('B','E'): 5,
        ('C','A'): 4, ('C','F'): 3,
        ('D','B'): 2,
        ('E','B'): 5, ('E','F'): 1,
        ('F','C'): 3, ('F','E'): 1
    }
    
    # Crear una instancia de la clase BusquedaGrafo.
    buscador = BusquedaGrafo(grafo_no_ponderado)
    
    # Nodo inicial y nodo objetivo.
    inicio = 'A'
    objetivo = 'F'
    
    # Ejecutar BFS.
    print("=== BFS ===")
    camino_bfs = buscador.bfs(inicio, objetivo)
    print(f"Camino BFS: {' → '.join(camino_bfs) if camino_bfs else 'No encontrado'}")
    
    # Ejecutar DFS.
    print("\n=== DFS ===")
    camino_dfs = buscador.dfs(inicio, objetivo)
    print(f"Camino DFS: {' → '.join(camino_dfs) if camino_dfs else 'No encontrado'}")
    
    # Ejecutar UCS.
    print("\n=== UCS ===")
    camino_ucs, costo_ucs = buscador.ucs(inicio, objetivo, costos)
    print(f"Camino UCS: {' → '.join(camino_ucs) if camino_ucs else 'No encontrado'}")
    print(f"Costo total: {costo_ucs if camino_ucs else 'N/A'}")