# Importamos deque de la biblioteca collections para usar una cola eficiente
# `deque` es una estructura de datos que permite agregar y quitar elementos de ambos extremos de manera eficiente.
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
    
    # Creamos una cola (FIFO) para almacenar los nodos pendientes por explorar.
    # `deque([inicio])` inicializa la cola con el nodo de inicio.
    cola = deque([inicio])
    
    # Diccionario para guardar el "padre" de cada nodo.
    # Esto nos permitirá reconstruir el camino desde el objetivo hasta el inicio.
    padres = {inicio: None}
    
    # Lista para registrar el orden en que se visitan los nodos.
    # Esto es útil para visualizar el proceso de búsqueda.
    orden_visita = []
    
    # Mientras haya nodos en la cola...
    while cola:
        # Sacamos el primer nodo de la cola (FIFO).
        # `popleft()` elimina y retorna el primer elemento de la cola.
        nodo_actual = cola.popleft()
        orden_visita.append(nodo_actual)  # Registramos el nodo visitado.
        
        # Si encontramos el nodo objetivo...
        if nodo_actual == objetivo:
            # Reconstruimos el camino desde el objetivo hasta el inicio.
            camino = []
            while nodo_actual is not None:
                camino.append(nodo_actual)  # Agregamos el nodo al camino.
                nodo_actual = padres[nodo_actual]  # Retrocedemos al nodo padre.
            # Invertimos el camino porque lo construimos desde el objetivo al inicio.
            return camino[::-1], orden_visita  # Retornamos el camino y el orden de visita.
        
        # Si no es el objetivo, exploramos sus vecinos.
        for vecino in grafo[nodo_actual]:
            # Si el vecino no ha sido visitado (no está en el diccionario `padres`).
            if vecino not in padres:
                # Marcamos al nodo actual como el padre del vecino.
                padres[vecino] = nodo_actual
                # Agregamos el vecino a la cola para explorarlo después.
                cola.append(vecino)
    
    # Si la cola se vacía y no encontramos el objetivo, retornamos None.
    return None, orden_visita

# **Ejemplo de uso**
if __name__ == "__main__":
    # Definimos un grafo como lista de adyacencia (diccionario).
    # Cada clave representa un nodo, y su valor es una lista de nodos vecinos.
    grafo_ejemplo = {
        'A': ['B', 'C'],    # A está conectado a B y C.
        'B': ['A', 'D', 'E'], # B está conectado a A, D y E.
        'C': ['A', 'F'],    # C está conectado a A y F.
        'D': ['B'],         # D solo está conectado a B.
        'E': ['B', 'F'],    # E está conectado a B y F.
        'F': ['C', 'E']     # F está conectado a C y E.
    }
    
    # Nodo inicial desde donde comienza la búsqueda.
    inicio = 'A'
    # Nodo objetivo que queremos encontrar.
    objetivo = 'F'
    
    # Ejecutamos la búsqueda en anchura (BFS).
    camino, orden_visita = busqueda_en_anchura(grafo_ejemplo, inicio, objetivo)
    
    # Mostramos los resultados.
    if camino:
        # Si se encontró un camino, lo mostramos.
        print(f"Camino encontrado de {inicio} a {objetivo}: {' -> '.join(camino)}")
        print(f"Orden de visita de nodos: {', '.join(orden_visita)}")
    else:
        # Si no se encontró un camino, lo indicamos.
        print(f"No se encontró un camino de {inicio} a {objetivo}")