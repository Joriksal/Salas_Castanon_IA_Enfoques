def busqueda_profundidad(grafo, inicio, objetivo):
    """
    Implementación de Búsqueda en Profundidad (DFS) para encontrar un camino
    desde el nodo inicial hasta el nodo objetivo en un grafo.
    
    Args:
        grafo (dict): Diccionario que representa el grafo (lista de adyacencia)
        inicio: Nodo inicial de la búsqueda
        objetivo: Nodo que queremos encontrar
    
    Returns:
        tuple: (camino_encontrado, nodos_visitados_en_orden)
               o (None, None) si no se encuentra el objetivo
    """
    # Inicializamos una pila (estructura LIFO: Last In, First Out) con el nodo inicial
    pila = [inicio]
    # Diccionario para registrar los padres de cada nodo (útil para reconstruir el camino)
    padres = {inicio: None}
    # Lista para registrar el orden en que se visitan los nodos
    orden_visita = []
    
    # Mientras haya nodos en la pila
    while pila:
        # Sacamos el último nodo agregado a la pila (LIFO)
        nodo_actual = pila.pop()
        # Registramos el nodo actual en el orden de visita
        orden_visita.append(nodo_actual)
        
        # Si el nodo actual es el objetivo, reconstruimos el camino desde el inicio
        if nodo_actual == objetivo:
            camino = []  # Lista para almacenar el camino encontrado
            while nodo_actual is not None:
                camino.append(nodo_actual)  # Agregamos el nodo al camino
                nodo_actual = padres[nodo_actual]  # Retrocedemos al nodo padre
            return camino[::-1], orden_visita  # Invertimos el camino para que sea de inicio a objetivo
        
        # Exploramos los vecinos del nodo actual en orden inverso
        # `reversed` invierte el orden de los vecinos para mantener el orden natural en la pila
        for vecino in reversed(grafo[nodo_actual]):
            # Si el vecino no ha sido visitado (no está en el diccionario de padres)
            if vecino not in padres:
                padres[vecino] = nodo_actual  # Registramos el nodo actual como padre del vecino
                pila.append(vecino)  # Agregamos el vecino a la pila para visitarlo más tarde
    
    # Si salimos del bucle, significa que no se encontró el objetivo
    return None, orden_visita

# Ejemplo de uso
if __name__ == "__main__":
    # Grafo representado como lista de adyacencia (diccionario)
    grafo_ejemplo = {
        'A': ['B', 'C'],  # El nodo 'A' tiene como vecinos a 'B' y 'C'
        'B': ['A', 'D', 'E'],  # El nodo 'B' tiene como vecinos a 'A', 'D' y 'E'
        'C': ['A', 'F'],  # El nodo 'C' tiene como vecinos a 'A' y 'F'
        'D': ['B'],  # El nodo 'D' tiene como vecino a 'B'
        'E': ['B', 'F'],  # El nodo 'E' tiene como vecinos a 'B' y 'F'
        'F': ['C', 'E']  # El nodo 'F' tiene como vecinos a 'C' y 'E'
    }
    
    # Nodo inicial de la búsqueda
    inicio = 'A'
    # Nodo objetivo que queremos encontrar
    objetivo = 'F'
    
    # Llamamos a la función de búsqueda en profundidad
    camino, orden_visita = busqueda_profundidad(grafo_ejemplo, inicio, objetivo)
    
    # Si se encontró un camino, lo imprimimos
    if camino:
        print(f"Camino encontrado de {inicio} a {objetivo}: {' -> '.join(camino)}")
        print(f"Orden de visita de nodos: {', '.join(orden_visita)}")
    else:
        # Si no se encontró un camino, lo indicamos
        print(f"No se encontró un camino de {inicio} a {objetivo}")