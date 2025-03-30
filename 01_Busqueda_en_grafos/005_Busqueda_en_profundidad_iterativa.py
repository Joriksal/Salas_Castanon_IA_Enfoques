def busqueda_profundidad_iterativa(grafo, inicio, objetivo, max_profundidad):
    """
    Implementación de Búsqueda en Profundidad Iterativa (IDS)
    
    Combina lo mejor de DFS (búsqueda en profundidad) y BFS (búsqueda en amplitud) 
    ejecutando múltiples DFS con límites incrementales.
    
    Args:
        grafo (dict): Diccionario de listas de adyacencia que representa el grafo.
        inicio (str): Nodo inicial desde donde comienza la búsqueda.
        objetivo (str): Nodo objetivo que se desea encontrar.
        max_profundidad (int): Máxima profundidad a considerar.
    
    Returns:
        tuple: (camino, nodos_visitados) si se encuentra el objetivo, o (None, nodos_visitados) si no se encuentra.
    """
    # Lista para registrar todos los nodos visitados en todas las iteraciones.
    todos_visitados = []
    
    # Ejecutar DLS (Búsqueda en Profundidad Limitada) con límites desde 0 hasta max_profundidad.
    for profundidad in range(max_profundidad + 1):
        # Ejecutar DLS con el límite actual.
        camino, visitados = dls(grafo, inicio, objetivo, profundidad)
        
        # Acumular los nodos visitados en esta iteración.
        todos_visitados.extend(visitados)
        
        # Si encontramos un camino al objetivo, retornar inmediatamente.
        if camino:
            return camino, todos_visitados
    
    # Si no se encuentra solución después de todas las iteraciones, retornar None y los nodos visitados.
    return None, todos_visitados

# Función auxiliar para realizar la Búsqueda en Profundidad Limitada (DLS).
def dls(grafo, inicio, objetivo, limite):
    """
    Implementación de Búsqueda en Profundidad Limitada (DLS).
    
    Args:
        grafo (dict): Diccionario de listas de adyacencia que representa el grafo.
        inicio (str): Nodo inicial desde donde comienza la búsqueda.
        objetivo (str): Nodo objetivo que se desea encontrar.
        limite (int): Profundidad máxima permitida para la búsqueda.
    
    Returns:
        tuple: (camino, nodos_visitados) si se encuentra el objetivo, o (None, nodos_visitados) si no se encuentra.
    """
    # Pila para almacenar los nodos a explorar. Cada elemento es una tupla (nodo, profundidad, camino).
    pila = [(inicio, 0, [inicio])]
    
    # Lista para registrar los nodos visitados en esta iteración.
    visitados = []
    
    # Mientras haya nodos en la pila para explorar.
    while pila:
        # Extraer el último nodo añadido a la pila (LIFO: Last In, First Out).
        nodo_actual, profundidad, camino = pila.pop()
        
        # Registrar el nodo actual como visitado.
        visitados.append(nodo_actual)
        
        # Verificar si el nodo actual es el objetivo.
        if nodo_actual == objetivo:
            return camino, visitados  # Retornar el camino encontrado y los nodos visitados.
            
        # Solo expandir los vecinos si no se ha alcanzado el límite de profundidad.
        if profundidad < limite:
            # Explorar los vecinos del nodo actual en orden inverso para mantener el orden natural.
            for vecino in reversed(grafo[nodo_actual]):
                # Evitar ciclos verificando que el vecino no esté ya en el camino actual.
                if vecino not in camino:
                    # Añadir el vecino a la pila con la nueva profundidad y el camino actualizado.
                    pila.append((vecino, profundidad + 1, camino + [vecino]))
    
    # Si se agotan los nodos en la pila sin encontrar el objetivo, retornar None y los nodos visitados.
    return None, visitados

# Ejemplo de uso
if __name__ == "__main__":
    # Definición de un grafo de ejemplo como un diccionario de listas de adyacencia.
    grafo = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    # Nodo inicial, nodo objetivo y profundidad máxima para la búsqueda.
    inicio = 'A'
    objetivo = 'F'
    max_prof = 3  # Máxima profundidad a considerar.
    
    # Ejecutar la búsqueda en profundidad iterativa.
    camino, visitados = busqueda_profundidad_iterativa(grafo, inicio, objetivo, max_prof)
    
    # Mostrar los resultados de la búsqueda.
    if camino:
        # Si se encuentra un camino, mostrarlo junto con la profundidad alcanzada.
        print(f"Camino encontrado: {' → '.join(camino)}")
        print(f"Profundidad necesaria: {len(camino)-1}")
    else:
        # Si no se encuentra un camino dentro del límite de profundidad, mostrar un mensaje.
        print("No se encontró solución dentro del límite")
    
    # Mostrar los nodos visitados en el orden en que fueron explorados.
    print(f"Nodos visitados: {visitados}")