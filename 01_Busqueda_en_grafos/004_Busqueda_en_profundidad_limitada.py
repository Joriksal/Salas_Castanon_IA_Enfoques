def busqueda_profundidad_limitada(grafo, inicio, objetivo, limite):
    """
    Implementación de Búsqueda en Profundidad Limitada (DLS)
    
    Parámetros:
        grafo (dict): Diccionario de listas de adyacencia que representa el grafo.
        inicio (str): Nodo inicial desde donde comienza la búsqueda.
        objetivo (str): Nodo objetivo que se desea encontrar.
        limite (int): Profundidad máxima permitida para la búsqueda.
    
    Retorna:
        tuple: (camino, nodos_visitados) si se encuentra el objetivo, o (None, nodos_visitados) si no se encuentra.
    """
    
    # Inicializa la pila (estructura LIFO) con el nodo inicial, profundidad 0 y el camino que contiene solo el nodo inicial.
    pila = [(inicio, 0, [inicio])]
    
    # Lista para registrar los nodos visitados en el orden en que se exploran.
    visitados = []
    
    # Mientras haya nodos en la pila para explorar.
    while pila:
        # Extraer el último nodo añadido a la pila (LIFO: Last In, First Out).
        nodo_actual, profundidad, camino = pila.pop()
        
        # Registrar el nodo actual como visitado.
        visitados.append(nodo_actual)
        
        # Verificar si el nodo actual es el objetivo.
        if nodo_actual == objetivo:
            # Si se encuentra el objetivo, retorna el camino y los nodos visitados.
            return camino, visitados
        
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
    grafo_ejemplo = {
        'A': ['B', 'C'],  # Nodo 'A' tiene como vecinos a 'B' y 'C'.
        'B': ['A', 'D', 'E'],  # Nodo 'B' tiene como vecinos a 'A', 'D' y 'E'.
        'C': ['A', 'F'],  # Nodo 'C' tiene como vecinos a 'A' y 'F'.
        'D': ['B'],  # Nodo 'D' tiene como vecino a 'B'.
        'E': ['B', 'F'],  # Nodo 'E' tiene como vecinos a 'B' y 'F'.
        'F': ['C', 'E']  # Nodo 'F' tiene como vecinos a 'C' y 'E'.
    }
    
    # Configuración inicial para la búsqueda.
    inicio = 'A'  # Nodo inicial desde donde comienza la búsqueda.
    objetivo = 'F'  # Nodo objetivo que se desea encontrar.
    limite_profundidad = 2  # Profundidad máxima permitida para la búsqueda.
    
    # Mensaje inicial.
    print("=== BÚSQUEDA EN PROFUNDIDAD LIMITADA ===")
    print(f"Buscando camino de {inicio} a {objetivo} con límite {limite_profundidad}")
    
    # Ejecutar la búsqueda en profundidad limitada.
    camino, visitados = busqueda_profundidad_limitada(grafo_ejemplo, inicio, objetivo, limite_profundidad)
    
    # Mostrar los resultados de la búsqueda.
    if camino:
        # Si se encuentra un camino, mostrarlo junto con la profundidad alcanzada.
        print("\n¡Camino encontrado!")
        print(f"Camino: {' → '.join(camino)}")  # Une los nodos del camino con una flecha.
        print(f"Profundidad alcanzada: {len(camino)-1}")  # Calcula la profundidad como el número de aristas.
    else:
        # Si no se encuentra un camino dentro del límite de profundidad, mostrar un mensaje.
        print("\nNo se encontró camino dentro del límite de profundidad")
    
    # Mostrar los nodos visitados en el orden en que fueron explorados.
    print(f"\nNodos visitados en orden: {', '.join(visitados)}")

