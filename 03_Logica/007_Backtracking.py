def es_seguro(tablero, fila, col, N):
    """
    Verifica si es seguro colocar una reina en la posición (fila, col).
    Una posición es segura si no hay otra reina en la misma columna,
    ni en las diagonales superior izquierda o superior derecha.
    """
    # Verificar la misma columna
    for i in range(fila):
        if tablero[i][col] == 1:
            return False  # Hay otra reina en la misma columna
    
    # Verificar la diagonal superior izquierda
    for i, j in zip(range(fila, -1, -1), range(col, -1, -1)):
        if tablero[i][j] == 1:
            return False  # Hay otra reina en la diagonal superior izquierda
    
    # Verificar la diagonal superior derecha
    for i, j in zip(range(fila, -1, -1), range(col, N)):
        if tablero[i][j] == 1:
            return False  # Hay otra reina en la diagonal superior derecha
    
    # Si no hay conflictos, la posición es segura
    return True

def resolver_n_reinas(tablero, fila, N):
    """
    Resuelve el problema de las N-Reinas utilizando backtracking.
    Intenta colocar reinas fila por fila, verificando que cada posición sea segura.
    
    Args:
        tablero: Matriz de NxN que representa el tablero de ajedrez.
        fila: Fila actual donde se intenta colocar una reina.
        N: Número de reinas (y tamaño del tablero).
    
    Returns:
        True si se encuentra una solución válida, False en caso contrario.
    """
    # Caso base: si todas las reinas están colocadas, se encontró una solución
    if fila >= N:
        return True
    
    # Intentar colocar una reina en cada columna de la fila actual
    for col in range(N):
        if es_seguro(tablero, fila, col, N):  # Verificar si es seguro colocar la reina
            tablero[fila][col] = 1  # Colocar la reina en la posición (fila, col)
            
            # Llamada recursiva para intentar colocar reinas en la siguiente fila
            if resolver_n_reinas(tablero, fila + 1, N):
                return True  # Si se encuentra una solución, retornar True
            
            # Si no se encuentra solución, hacer backtracking (quitar la reina)
            tablero[fila][col] = 0
    
    # Si no se puede colocar una reina en ninguna columna de esta fila, retornar False
    return False

# Ejemplo de uso
if __name__ == "__main__":
    N = 4  # Tamaño del tablero y número de reinas
    # Crear un tablero vacío de NxN (matriz inicializada con ceros)
    tablero = [[0] * N for _ in range(N)]
    
    # Intentar resolver el problema de las N-Reinas
    if resolver_n_reinas(tablero, 0, N):
        # Si se encuentra una solución, imprimir el tablero resultante
        for fila in tablero:
            print(fila)
    else:
        # Si no hay solución, imprimir un mensaje indicándolo
        print("No hay solución")