from collections import deque
import numpy as np

# ----------------------------- 8-Puzzle -----------------------------
class Puzzle8:
    """Clase que representa el espacio de estados para el problema del 8-Puzzle."""
    
    def __init__(self, initial_state, goal_state):
        """
        Inicializa el estado inicial y el estado objetivo.
        :param initial_state: Matriz 3x3 representando el estado inicial.
        :param goal_state: Matriz 3x3 representando el estado objetivo.
        """
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)
    
    def find_empty(self, state):
        """
        Encuentra la posición del espacio vacío (representado por 0) en el estado actual.
        :param state: Matriz 3x3 representando el estado actual.
        :return: Tupla con las coordenadas (fila, columna) del espacio vacío.
        """
        return tuple(np.argwhere(state == 0)[0])
    
    def get_actions(self, state):
        """
        Genera las acciones posibles para mover el espacio vacío.
        :param state: Matriz 3x3 representando el estado actual.
        :return: Lista de acciones posibles (arriba, abajo, izquierda, derecha).
        """
        empty_row, empty_col = self.find_empty(state)
        actions = []
        if empty_row > 0:
            actions.append("Mover arriba")
        if empty_row < 2:
            actions.append("Mover abajo")
        if empty_col > 0:
            actions.append("Mover izquierda")
        if empty_col < 2:
            actions.append("Mover derecha")
        return actions
    
    def apply_action(self, state, action):
        """
        Aplica una acción al estado actual para generar un nuevo estado.
        :param state: Matriz 3x3 representando el estado actual.
        :param action: Acción a aplicar (arriba, abajo, izquierda, derecha).
        :return: Nuevo estado después de aplicar la acción.
        """
        new_state = state.copy()
        row, col = self.find_empty(new_state)
        if action == "Mover arriba":
            new_state[row, col], new_state[row-1, col] = new_state[row-1, col], new_state[row, col]
        elif action == "Mover abajo":
            new_state[row, col], new_state[row+1, col] = new_state[row+1, col], new_state[row, col]
        elif action == "Mover izquierda":
            new_state[row, col], new_state[row, col-1] = new_state[row, col-1], new_state[row, col]
        elif action == "Mover derecha":
            new_state[row, col], new_state[row, col+1] = new_state[row, col+1], new_state[row, col]
        return new_state
    
    def solve_bfs(self):
        """
        Resuelve el problema del 8-Puzzle utilizando búsqueda en anchura (BFS).
        :return: Lista de acciones para llegar al estado objetivo o None si no hay solución.
        """
        queue = deque([(self.initial_state, [])])  # Cola para BFS con el estado inicial y el camino vacío.
        visited = set()  # Conjunto para rastrear estados visitados.
        visited.add(tuple(map(tuple, self.initial_state)))  # Convertir el estado inicial a una tupla inmutable.
        
        while queue:
            current_state, path = queue.popleft()  # Extraer el estado actual y el camino recorrido.
            if np.array_equal(current_state, self.goal_state):  # Verificar si se alcanzó el estado objetivo.
                return path
            
            for action in self.get_actions(current_state):  # Generar acciones posibles.
                new_state = self.apply_action(current_state, action)  # Aplicar acción para obtener nuevo estado.
                new_state_tuple = tuple(map(tuple, new_state))  # Convertir a tupla para verificar en "visited".
                if new_state_tuple not in visited:  # Si no se ha visitado, agregar a la cola y marcar como visitado.
                    visited.add(new_state_tuple)
                    queue.append((new_state, path + [action]))
        return None  # Retornar None si no hay solución.

# ----------------------------- Laberinto -----------------------------
def solve_maze(maze, start, end):
    """
    Resuelve un laberinto utilizando búsqueda en anchura (BFS).
    :param maze: Matriz 2D representando el laberinto ('.' para camino, '#' para pared).
    :param start: Tupla con las coordenadas iniciales (x, y).
    :param end: Tupla con las coordenadas finales (x, y).
    :return: Lista de acciones para llegar al final o None si no hay solución.
    """
    rows, cols = len(maze), len(maze[0])  # Dimensiones del laberinto.
    queue = deque([(start, [])])  # Cola para BFS con la posición inicial y el camino vacío.
    visited = set()  # Conjunto para rastrear posiciones visitadas.
    visited.add(start)  # Marcar la posición inicial como visitada.
    
    while queue:
        (x, y), path = queue.popleft()  # Extraer la posición actual y el camino recorrido.
        if (x, y) == end:  # Verificar si se alcanzó la posición final.
            return path
        
        # Generar movimientos posibles: arriba, abajo, izquierda, derecha.
        for dx, dy, action in [(-1, 0, "Arriba"), (1, 0, "Abajo"), (0, -1, "Izquierda"), (0, 1, "Derecha")]:
            nx, ny = x + dx, y + dy  # Calcular nueva posición.
            # Verificar si la nueva posición es válida y no visitada.
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != "#" and (nx, ny) not in visited:
                visited.add((nx, ny))  # Marcar como visitada.
                queue.append(((nx, ny), path + [action]))  # Agregar nueva posición y camino a la cola.
    return None  # Retornar None si no hay solución.

# ----------------------------- Ejemplos de Uso -----------------------------
if __name__ == "__main__":
    # Ejemplo del 8-Puzzle
    print("----- 8-Puzzle -----")
    puzzle = Puzzle8(
        initial_state=[[1, 2, 3], [4, 0, 5], [7, 8, 6]],  # Estado inicial.
        goal_state=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # Estado objetivo.
    )
    solution_puzzle = puzzle.solve_bfs()
    print("Solución 8-Puzzle:", solution_puzzle)

    # Ejemplo del Laberinto
    print("\n----- Laberinto -----")
    maze = [
        [".", "#", ".", ".", "."],
        [".", "#", ".", "#", "."],
        [".", ".", ".", ".", "."],
        ["#", "#", ".", "#", "."],
        [".", ".", ".", "#", "."]
    ]
    start = (0, 0)  # Posición inicial.
    end = (4, 4)  # Posición final.
    solution_maze = solve_maze(maze, start, end)
    print("Solución Laberinto:", solution_maze)