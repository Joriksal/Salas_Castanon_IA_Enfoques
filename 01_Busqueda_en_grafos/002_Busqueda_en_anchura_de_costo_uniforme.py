# Importamos heapq para implementar una cola de prioridad eficiente
import heapq

def busqueda_costo_uniforme(grafo, inicio, objetivo):
    """
    Implementación de Búsqueda de Costo Uniforme (UCS) para encontrar
    el camino de menor costo en un grafo con pesos no negativos.
    
    Args:
        grafo (dict): Diccionario de listas de adyacencia con formato
                     {nodo: [(vecino1, costo1), (vecino2, costo2), ...]}
        inicio (hashable): Nodo inicial de la búsqueda
        objetivo (hashable): Nodo objetivo a encontrar
    
    Returns:
        tuple: (costo_total, lista_camino) si se encuentra el objetivo,
               (None, None) si no existe camino
    """
    
    # Inicializamos la cola de prioridad con el nodo inicial
    # Cada elemento es una tupla (costo_acumulado, nodo_actual, camino)
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0, inicio, [inicio]))
    
    # Diccionario para mantener registro de los costos mínimos encontrados
    # hasta cada nodo. Evita reprocesamiento innecesario.
    costos_minimos = {inicio: 0}
    
    # Mientras haya nodos por explorar en la cola
    while cola_prioridad:
        # Extraemos el nodo con menor costo acumulado (propiedad del heap)
        costo_acumulado, nodo_actual, camino = heapq.heappop(cola_prioridad)
        
        # Poda: si encontramos un camino mejor a este nodo previamente,
        # ignoramos este camino más costoso
        if costo_acumulado > costos_minimos.get(nodo_actual, float('inf')):
            continue
        
        # Si llegamos al nodo objetivo, retornamos el resultado
        if nodo_actual == objetivo:
            return (costo_acumulado, camino)
        
        # Exploramos todos los vecinos del nodo actual
        for vecino, costo in grafo[nodo_actual]:
            # Calculamos el nuevo costo acumulado
            nuevo_costo = costo_acumulado + costo
            
            # Solo consideramos este camino si es mejor que los encontrados antes
            if vecino not in costos_minimos or nuevo_costo < costos_minimos[vecino]:
                # Actualizamos el costo mínimo para este vecino
                costos_minimos[vecino] = nuevo_costo
                # Agregamos a la cola con el nuevo camino
                heapq.heappush(
                    cola_prioridad, 
                    (nuevo_costo, vecino, camino + [vecino])
                )
    
    # Si la cola se vacía sin encontrar el objetivo
    return (None, None)


# Ejemplo de grafo ponderado
grafo_con_pesos = {
    'A': [('B', 1), ('C', 4)],  # A conecta a B (costo 1) y C (costo 4)
    'B': [('A', 1), ('D', 2), ('E', 5)],  # B conecta a A, D y E
    'C': [('A', 4), ('F', 3)],  # C conecta a A y F
    'D': [('B', 2)],  # D solo conecta a B
    'E': [('B', 5), ('F', 1)],  # E conecta a B y F
    'F': [('C', 3), ('E', 1)]  # F conecta a C y E
}

# Ejemplo de uso
if __name__ == "__main__":
    # Definimos inicio y objetivo
    nodo_inicio = 'A'
    nodo_objetivo = 'F'
    
    # Ejecutamos UCS
    costo, camino = busqueda_costo_uniforme(grafo_con_pesos, nodo_inicio, nodo_objetivo)
    
    # Mostramos resultados
    if camino:
        print(f"Camino encontrado: {' -> '.join(camino)}")
        print(f"Costo total del camino: {costo}")
        print("\nDesglose del costo:")
        for i in range(len(camino)-1):
            nodo_actual = camino[i]
            nodo_siguiente = camino[i+1]
            # Buscamos el costo entre estos nodos en el grafo
            costo_arista = next(c for (n, c) in grafo_con_pesos[nodo_actual] if n == nodo_siguiente)
            print(f"{nodo_actual} -> {nodo_siguiente}: {costo_arista}")
    else:
        print("No se encontró un camino válido al objetivo.")