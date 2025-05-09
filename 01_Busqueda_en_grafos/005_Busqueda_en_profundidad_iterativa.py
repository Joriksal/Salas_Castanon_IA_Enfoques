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
    
    # Bucle que ejecuta DLS (Búsqueda en Profundidad Limitada) con límites desde 0 hasta max_profundidad.
    for profundidad in range(max_profundidad + 1):  # range genera números desde 0 hasta max_profundidad (inclusive).
        # Ejecutar DLS con el límite actual.
        camino, visitados = dls(grafo, inicio, objetivo, profundidad)
        
        # Acumular los nodos visitados en esta iteración.
        todos_visitados.extend(visitados)  # extend añade todos los elementos de visitados a todos_visitados.
        
        # Si encontramos un camino al objetivo, retornar inmediatamente.
        if camino:  # Verifica si camino no es None (es decir, si se encontró un camino).
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
    pila = [(inicio, 0, [inicio])]  # La pila comienza con el nodo inicial, profundidad 0 y un camino que contiene solo el nodo inicial.
    
    # Lista para registrar los nodos visitados en esta iteración.
    visitados = []
    
    # Mientras haya nodos en la pila para explorar.
    while pila:  # while ejecuta el bloque mientras la pila no esté vacía.
        # Extraer el último nodo añadido a la pila (LIFO: Last In, First Out).
        nodo_actual, profundidad, camino = pila.pop()  # pop elimina y retorna el último elemento de la pila.
        
        # Registrar el nodo actual como visitado.
        visitados.append(nodo_actual)  # append añade el nodo actual a la lista visitados.
        
        # Verificar si el nodo actual es el objetivo.
        if nodo_actual == objetivo:  # Compara si el nodo actual es igual al nodo objetivo.
            return camino, visitados  # Retornar el camino encontrado y los nodos visitados.
            
        # Solo expandir los vecinos si no se ha alcanzado el límite de profundidad.
        if profundidad < limite:  # Verifica si la profundidad actual es menor que el límite.
            # Explorar los vecinos del nodo actual en orden inverso para mantener el orden natural.
            for vecino in reversed(grafo[nodo_actual]):  # reversed invierte el orden de los vecinos.
                # Evitar ciclos verificando que el vecino no esté ya en el camino actual.
                if vecino not in camino:  # Verifica si el vecino no está en el camino actual para evitar ciclos.
                    # Añadir el vecino a la pila con la nueva profundidad y el camino actualizado.
                    pila.append((vecino, profundidad + 1, camino + [vecino]))  # Crea una nueva tupla con el vecino, profundidad incrementada y el camino actualizado.
    
    # Si se agotan los nodos en la pila sin encontrar el objetivo, retornar None y los nodos visitados.
    return None, visitados

# Ejemplo de uso
if __name__ == "__main__":  # Verifica si el archivo se está ejecutando directamente (no importado como módulo).
    # Definición de un grafo de ejemplo como un diccionario de listas de adyacencia.
    grafo = {
        'A': ['B', 'C'],  # El nodo 'A' tiene como vecinos a 'B' y 'C'.
        'B': ['A', 'D', 'E'],  # El nodo 'B' tiene como vecinos a 'A', 'D' y 'E'.
        'C': ['A', 'F'],  # El nodo 'C' tiene como vecinos a 'A' y 'F'.
        'D': ['B'],  # El nodo 'D' tiene como vecino a 'B'.
        'E': ['B', 'F'],  # El nodo 'E' tiene como vecinos a 'B' y 'F'.
        'F': ['C', 'E']  # El nodo 'F' tiene como vecinos a 'C' y 'E'.
    }
    
    # Nodo inicial, nodo objetivo y profundidad máxima para la búsqueda.
    inicio = 'A'  # Nodo desde donde comienza la búsqueda.
    objetivo = 'F'  # Nodo que se desea encontrar.
    max_prof = 3  # Máxima profundidad a considerar.
    
    # Ejecutar la búsqueda en profundidad iterativa.
    camino, visitados = busqueda_profundidad_iterativa(grafo, inicio, objetivo, max_prof)
    
    # Mostrar los resultados de la búsqueda.
    if camino:  # Si se encontró un camino.
        # Si se encuentra un camino, mostrarlo junto con la profundidad alcanzada.
        print(f"Camino encontrado: {' → '.join(camino)}")  # Une los nodos del camino con flechas.
        print(f"Profundidad necesaria: {len(camino)-1}")  # Calcula la profundidad como el número de aristas en el camino.
    else:  # Si no se encontró un camino dentro del límite de profundidad.
        print("No se encontró solución dentro del límite")
    
    # Mostrar los nodos visitados en el orden en que fueron explorados.
    print(f"Nodos visitados: {visitados}")  # Muestra la lista de nodos visitados.