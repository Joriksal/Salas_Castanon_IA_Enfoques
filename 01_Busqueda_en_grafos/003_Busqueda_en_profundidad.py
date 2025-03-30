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
    # Pila para nodos por visitar (LIFO)
    pila = [inicio]
    # Diccionario para registrar los padres de cada nodo
    padres = {inicio: None}
    # Lista para registrar el orden de visita
    orden_visita = []
    
    while pila:
        # Sacamos el último nodo agregado (LIFO)
        nodo_actual = pila.pop()
        orden_visita.append(nodo_actual)
        
        # Si encontramos el objetivo, reconstruimos el camino
        if nodo_actual == objetivo:
            camino = []
            while nodo_actual is not None:
                camino.append(nodo_actual)
                nodo_actual = padres[nodo_actual]
            return camino[::-1], orden_visita  # Invertimos el camino
        
        # Exploramos los vecinos no visitados en orden inverso
        # para mantener el orden natural en la pila
        for vecino in reversed(grafo[nodo_actual]):
            if vecino not in padres:
                padres[vecino] = nodo_actual
                pila.append(vecino)
    
    # Si llegamos aquí, no se encontró el objetivo
    return None, orden_visita

# Ejemplo de uso
if __name__ == "__main__":
    # Grafo representado como lista de adyacencia
    grafo_ejemplo = {
        'A': ['B', 'C'],
        'B': ['A', 'D', 'E'],
        'C': ['A', 'F'],
        'D': ['B'],
        'E': ['B', 'F'],
        'F': ['C', 'E']
    }
    
    inicio = 'A'
    objetivo = 'F'
    
    camino, orden_visita = busqueda_profundidad(grafo_ejemplo, inicio, objetivo)
    
    if camino:
        print(f"Camino encontrado de {inicio} a {objetivo}: {' -> '.join(camino)}")
        print(f"Orden de visita de nodos: {', '.join(orden_visita)}")
    else:
        print(f"No se encontró un camino de {inicio} a {objetivo}")