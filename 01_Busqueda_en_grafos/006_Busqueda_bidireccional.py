from collections import deque  # Importación de deque para implementar colas eficientes.

def busqueda_bidireccional(grafo, inicio, objetivo):
    """
    Implementación de Búsqueda Bidireccional usando BFS desde ambos extremos.
    
    Args:
        grafo (dict): Diccionario de listas de adyacencia que representa el grafo.
        inicio (str): Nodo inicial desde donde comienza la búsqueda.
        objetivo (str): Nodo objetivo que se desea encontrar.
    
    Returns:
        tuple: (camino, nodos_visitados) si se encuentra un camino, o (None, None) si no hay camino.
    """
    # Verificación rápida si el nodo inicial y el objetivo son el mismo.
    if inicio == objetivo:
        return [inicio], [inicio]
    
    # Estructuras para la búsqueda desde el nodo inicial.
    cola_inicio = deque([inicio])  # Cola para BFS desde el inicio.
    padres_inicio = {inicio: None}  # Diccionario para rastrear los padres de cada nodo.
    visitados_inicio = set([inicio])  # Conjunto de nodos visitados desde el inicio.
    
    # Estructuras para la búsqueda desde el nodo objetivo.
    cola_objetivo = deque([objetivo])  # Cola para BFS desde el objetivo.
    padres_objetivo = {objetivo: None}  # Diccionario para rastrear los padres de cada nodo.
    visitados_objetivo = set([objetivo])  # Conjunto de nodos visitados desde el objetivo.
    
    # Variable para registrar el punto de intersección entre las búsquedas.
    interseccion = None
    
    # Mientras ambas colas tengan nodos por explorar.
    while cola_inicio and cola_objetivo:
        # Búsqueda desde el inicio.
        nodo_actual_inicio = cola_inicio.popleft()  # Extraer el nodo actual desde la cola.
        for vecino in grafo[nodo_actual_inicio]:  # Explorar los vecinos del nodo actual.
            if vecino not in padres_inicio:  # Si el vecino no ha sido visitado desde el inicio.
                padres_inicio[vecino] = nodo_actual_inicio  # Registrar el padre del vecino.
                cola_inicio.append(vecino)  # Añadir el vecino a la cola.
                visitados_inicio.add(vecino)  # Marcar el vecino como visitado.
                
                # Comprobar si hay intersección con los nodos visitados desde el objetivo.
                if vecino in visitados_objetivo:
                    interseccion = vecino  # Registrar el nodo de intersección.
                    break  # Salir del bucle si se encuentra intersección.
        
        if interseccion:  # Si se encontró intersección, salir del bucle principal.
            break
            
        # Búsqueda desde el objetivo.
        nodo_actual_objetivo = cola_objetivo.popleft()  # Extraer el nodo actual desde la cola.
        for vecino in grafo[nodo_actual_objetivo]:  # Explorar los vecinos del nodo actual.
            if vecino not in padres_objetivo:  # Si el vecino no ha sido visitado desde el objetivo.
                padres_objetivo[vecino] = nodo_actual_objetivo  # Registrar el padre del vecino.
                cola_objetivo.append(vecino)  # Añadir el vecino a la cola.
                visitados_objetivo.add(vecino)  # Marcar el vecino como visitado.
                
                # Comprobar si hay intersección con los nodos visitados desde el inicio.
                if vecino in visitados_inicio:
                    interseccion = vecino  # Registrar el nodo de intersección.
                    break  # Salir del bucle si se encuentra intersección.
        
        if interseccion:  # Si se encontró intersección, salir del bucle principal.
            break
    
    # Reconstruir el camino si hay intersección.
    if interseccion:
        # Construir el camino desde el inicio hasta la intersección.
        camino_inicio = []
        nodo = interseccion
        while nodo is not None:  # Seguir los padres hasta llegar al nodo inicial.
            camino_inicio.append(nodo)
            nodo = padres_inicio[nodo]
        camino_inicio.reverse()  # Invertir el camino para que vaya del inicio a la intersección.
        
        # Construir el camino desde la intersección hasta el objetivo (excluyendo la intersección).
        camino_objetivo = []
        nodo = padres_objetivo[interseccion]
        while nodo is not None:  # Seguir los padres hasta llegar al nodo objetivo.
            camino_objetivo.append(nodo)
            nodo = padres_objetivo[nodo]
        
        # Combinar los caminos para obtener el camino completo.
        camino_completo = camino_inicio + camino_objetivo
        # Unir los nodos visitados desde ambas búsquedas.
        todos_visitados = list(visitados_inicio.union(visitados_objetivo))
        
        return camino_completo, todos_visitados  # Retornar el camino y los nodos visitados.
    
    # Si no se encontró intersección, no hay camino entre los nodos.
    return None, None

# Ejemplo de uso
if __name__ == "__main__":
    # Grafo de ejemplo representado como un diccionario de listas de adyacencia.
    grafo = {
        'A': ['B', 'C'],  # Nodo A conectado a B y C.
        'B': ['A', 'D', 'E'],  # Nodo B conectado a A, D y E.
        'C': ['A', 'F'],  # Nodo C conectado a A y F.
        'D': ['B'],  # Nodo D conectado a B.
        'E': ['B', 'F'],  # Nodo E conectado a B y F.
        'F': ['C', 'E', 'G'],  # Nodo F conectado a C, E y G.
        'G': ['F']  # Nodo G conectado a F.
    }
    
    # Nodo inicial y nodo objetivo.
    inicio = 'A'  # Nodo desde donde comienza la búsqueda.
    objetivo = 'G'  # Nodo que se desea encontrar.
    
    # Ejecutar la búsqueda bidireccional.
    camino, visitados = busqueda_bidireccional(grafo, inicio, objetivo)
    
    # Mostrar los resultados de la búsqueda.
    if camino:
        # Si se encuentra un camino, mostrarlo junto con los nodos visitados.
        print(f"Camino encontrado: {' → '.join(camino)}")  # Camino encontrado.
        print(f"Nodos visitados: {len(visitados)}")  # Número total de nodos visitados.
        print(f"Todos los nodos visitados: {', '.join(sorted(visitados))}")  # Lista de nodos visitados.
    else:
        # Si no se encuentra un camino, mostrar un mensaje.
        print("No se encontró camino entre los nodos")  # Mensaje de error.