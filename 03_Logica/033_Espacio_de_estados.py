# Importamos la clase deque de la librería collections para manejar una cola eficiente.
from collections import deque
# Importamos numpy para trabajar con matrices de manera más sencilla.
import numpy as np

# ----------------------------- 8-Puzzle -----------------------------
class Puzzle8:
    """Clase que representa el espacio de estados para el problema del 8-Puzzle."""

    def __init__(self, initial_state, goal_state):
        """
        Inicializa el estado inicial y el estado objetivo del 8-Puzzle.
        :param initial_state: Matriz 3x3 representando el estado inicial del puzzle.
        :param goal_state: Matriz 3x3 representando el estado objetivo del puzzle.
        """
        # Convertimos los estados inicial y objetivo a matrices de numpy para facilitar las operaciones.
        self.initial_state = np.array(initial_state)
        self.goal_state = np.array(goal_state)

    def find_empty(self, state):
        """
        Encuentra la posición del espacio vacío (representado por el número 0) en el estado actual.
        :param state: Matriz 3x3 representando el estado actual del puzzle.
        :return: Tupla con las coordenadas (fila, columna) del espacio vacío.
        """
        # Utilizamos np.argwhere para encontrar la posición del valor 0 en la matriz.
        return tuple(np.argwhere(state == 0)[0])

    def get_actions(self, state):
        """
        Genera las acciones posibles para mover el espacio vacío en el puzzle.
        :param state: Matriz 3x3 representando el estado actual del puzzle.
        :return: Lista de acciones posibles (arriba, abajo, izquierda, derecha).
        """
        # Obtenemos la posición del espacio vacío.
        empty_row, empty_col = self.find_empty(state)
        actions = []  # Lista para almacenar las acciones posibles.

        # Verificamos si es posible mover el espacio vacío hacia arriba.
        if empty_row > 0:
            actions.append("Mover arriba")
        # Verificamos si es posible mover el espacio vacío hacia abajo.
        if empty_row < 2:
            actions.append("Mover abajo")
        # Verificamos si es posible mover el espacio vacío hacia la izquierda.
        if empty_col > 0:
            actions.append("Mover izquierda")
        # Verificamos si es posible mover el espacio vacío hacia la derecha.
        if empty_col < 2:
            actions.append("Mover derecha")

        return actions

    def apply_action(self, state, action):
        """
        Aplica una acción al estado actual para generar un nuevo estado.
        :param state: Matriz 3x3 representando el estado actual del puzzle.
        :param action: Acción a aplicar (arriba, abajo, izquierda, derecha).
        :return: Nuevo estado después de aplicar la acción.
        """
        # Creamos una copia del estado actual para no modificar el original.
        new_state = state.copy()
        # Obtenemos la posición del espacio vacío.
        row, col = self.find_empty(new_state)

        # Intercambiamos posiciones según la acción especificada.
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
        # Inicializamos la cola para BFS con el estado inicial y un camino vacío.
        queue = deque([(self.initial_state, [])])
        # Conjunto para rastrear los estados visitados y evitar ciclos.
        visited = set()
        # Convertimos el estado inicial a una tupla inmutable y lo marcamos como visitado.
        visited.add(tuple(map(tuple, self.initial_state)))

        # Mientras haya elementos en la cola, seguimos explorando.
        while queue:
            # Extraemos el estado actual y el camino recorrido hasta ahora.
            current_state, path = queue.popleft()

            # Si el estado actual es igual al estado objetivo, retornamos el camino.
            if np.array_equal(current_state, self.goal_state):
                return path

            # Generamos todas las acciones posibles desde el estado actual.
            for action in self.get_actions(current_state):
                # Aplicamos la acción para obtener un nuevo estado.
                new_state = self.apply_action(current_state, action)
                # Convertimos el nuevo estado a una tupla para verificar si ya fue visitado.
                new_state_tuple = tuple(map(tuple, new_state))

                # Si el nuevo estado no ha sido visitado, lo agregamos a la cola y lo marcamos como visitado.
                if new_state_tuple not in visited:
                    visited.add(new_state_tuple)
                    queue.append((new_state, path + [action]))

        # Si agotamos la cola y no encontramos solución, retornamos None.
        return None

# ----------------------------- Laberinto -----------------------------
def solve_maze(maze, start, end):
    """
    Resuelve un laberinto utilizando búsqueda en anchura (BFS).
    :param maze: Matriz 2D representando el laberinto ('.' para camino, '#' para pared).
    :param start: Tupla con las coordenadas iniciales (x, y).
    :param end: Tupla con las coordenadas finales (x, y).
    :return: Lista de acciones para llegar al final o None si no hay solución.
    """
    # Obtenemos las dimensiones del laberinto.
    rows, cols = len(maze), len(maze[0])
    # Inicializamos la cola para BFS con la posición inicial y un camino vacío.
    queue = deque([(start, [])])
    # Conjunto para rastrear las posiciones visitadas.
    visited = set()
    # Marcamos la posición inicial como visitada.
    visited.add(start)

    # Mientras haya elementos en la cola, seguimos explorando.
    while queue:
        # Extraemos la posición actual y el camino recorrido hasta ahora.
        (x, y), path = queue.popleft()

        # Si alcanzamos la posición final, retornamos el camino.
        if (x, y) == end:
            return path

        # Generamos los movimientos posibles: arriba, abajo, izquierda, derecha.
        for dx, dy, action in [(-1, 0, "Arriba"), (1, 0, "Abajo"), (0, -1, "Izquierda"), (0, 1, "Derecha")]:
            # Calculamos la nueva posición.
            nx, ny = x + dx, y + dy

            # Verificamos si la nueva posición es válida y no ha sido visitada.
            if 0 <= nx < rows and 0 <= ny < cols and maze[nx][ny] != "#" and (nx, ny) not in visited:
                # Marcamos la nueva posición como visitada.
                visited.add((nx, ny))
                # Agregamos la nueva posición y el camino actualizado a la cola.
                queue.append(((nx, ny), path + [action]))

    # Si agotamos la cola y no encontramos solución, retornamos None.
    return None

# ----------------------------- Ejemplos de Uso -----------------------------
if __name__ == "__main__":
    # Ejemplo del 8-Puzzle
    print("----- 8-Puzzle -----")
    puzzle = Puzzle8(
        initial_state=[[1, 2, 3], [4, 0, 5], [7, 8, 6]],  # Estado inicial del puzzle.
        goal_state=[[1, 2, 3], [4, 5, 6], [7, 8, 0]]  # Estado objetivo del puzzle.
    )
    # Resolvemos el 8-Puzzle utilizando BFS.
    solution_puzzle = puzzle.solve_bfs()
    print("Solución 8-Puzzle:", solution_puzzle)

    # Ejemplo del Laberinto
    print("\n----- Laberinto -----")
    maze = [
        [".", "#", ".", ".", "."],  # Representación del laberinto.
        [".", "#", ".", "#", "."],
        [".", ".", ".", ".", "."],
        ["#", "#", ".", "#", "."],
        [".", ".", ".", "#", "."]
    ]
    start = (0, 0)  # Posición inicial en el laberinto.
    end = (4, 4)  # Posición final en el laberinto.
    # Resolvemos el laberinto utilizando BFS.
    solution_maze = solve_maze(maze, start, end)
    print("Solución Laberinto:", solution_maze)