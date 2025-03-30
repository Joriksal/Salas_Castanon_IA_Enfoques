# Importamos deque de la biblioteca collections para usar una cola eficiente
from collections import deque

def busqueda_en_anchura(grafo, inicio, objetivo):
    """  
    Este algoritmo explora todos los nodos nivel por nivel, garantizando encontrar
    el camino más corto en un grafo no ponderado.
    
    Parámetros:
        grafo (dict): Representación del grafo como lista de adyacencia.
                      Ejemplo: {'A': ['B', 'C'], 'B': ['A', 'D'], ...}
        inicio: Nodo desde donde comienza la búsqueda.
        objetivo: Nodo que queremos encontrar.
    
    Retorna:
        tuple: (camino_encontrado, nodos_visitados_en_orden)
               Si no encuentra el objetivo, retorna (None, nodos_visitados)
    """
    
    # Usamos una cola (FIFO) para almacenar los nodos pendientes por explorar
    cola = deque([inicio])
    
    # Diccionario para guardar el "padre" de cada nodo y así reconstruir el camino
    padres = {inicio: None}
    
    # Lista para guardar el orden en que se visitan los nodos (útil para visualización)
    orden_visita = []
    
    # Mientras haya nodos en la cola...
    while cola:
        # Sacamos el primer nodo de la cola (FIFO)
        nodo_actual = cola.popleft()
        orden_visita.append(nodo_actual)
        
        # Si encontramos el objetivo, reconstruimos el camino desde el final
        if nodo_actual == objetivo:
            camino = []
            # Retrocedemos desde el objetivo hasta el inicio usando los padres
            while nodo_actual is not None:
                camino.append(nodo_actual)
                nodo_actual = padres[nodo_actual]
            # El camino está en orden inverso, así que lo invertimos
            return camino[::-1], orden_visita
        
        # Si no es el objetivo, exploramos sus vecinos
        for vecino in grafo[nodo_actual]:
            # Si el vecino no ha sido visitado (no está en 'padres')
            if vecino not in padres:
                # Marcamos al nodo actual como su padre
                padres[vecino] = nodo_actual
                # Lo agregamos a la cola para explorarlo después
                cola.append(vecino)
    
    # Si la cola se vacía y no se encontró el objetivo, retornamos None
    return None, orden_visita

# **Ejemplo de uso**
if __name__ == "__main__":
    # Definimos un grafo como lista de adyacencia (diccionario)
    grafo_ejemplo = {
        'A': ['B', 'C'],    # A está conectado a B y C
        'B': ['A', 'D', 'E'], # B está conectado a A, D y E
        'C': ['A', 'F'],    # C está conectado a A y F
        'D': ['B'],         # D solo está conectado a B
        'E': ['B', 'F'],    # E está conectado a B y F
        'F': ['C', 'E']     # F está conectado a C y E
    }
    
    inicio = 'A'  # Nodo inicial
    objetivo = 'F'  # Nodo objetivo
    
    # Ejecutamos BFS
    camino, orden_visita = busqueda_en_anchura(grafo_ejemplo, inicio, objetivo)
    
    # Mostramos resultados
    if camino:
        print(f"Camino encontrado de {inicio} a {objetivo}: {' -> '.join(camino)}")
        print(f"Orden de visita de nodos: {', '.join(orden_visita)}")
    else:
        print(f"No se encontró un camino de {inicio} a {objetivo}")