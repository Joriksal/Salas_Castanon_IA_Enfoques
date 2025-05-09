from collections import deque
# Importa la clase deque (double-ended queue) del módulo collections.
# deque proporciona una estructura de datos similar a una lista con adiciones y eliminaciones eficientes en ambos extremos.

class GameGraph:
    def __init__(self, payoff_matrix):
        """
        Inicializa el grafo del juego con una matriz de pagos.

        Args:
            payoff_matrix (dict): Matriz de pagos para cada jugador.
                Formato: {(estrategia_j1, estrategia_j2): (pago_j1, pago_j2)}
                - La clave es una tupla de las estrategias elegidas por el jugador 1 y el jugador 2.
                - El valor es una tupla de los pagos correspondientes para el jugador 1 y el jugador 2.
        """
        self.payoff_matrix = payoff_matrix
        # Almacena la matriz de pagos proporcionada.
        self.strategies = self._get_unique_strategies()  # Extraer estrategias únicas del juego.
        # Llama al método _get_unique_strategies para obtener una lista ordenada de todas las estrategias posibles en el juego.

    def _get_unique_strategies(self):
        """
        Extrae las estrategias únicas de la matriz de pagos.

        Returns:
            list: Lista de estrategias únicas, ordenada alfabéticamente.
        """
        strategies = set()
        # Inicializa un conjunto vacío llamado 'strategies'. Los conjuntos almacenan solo elementos únicos.
        for s1, s2 in self.payoff_matrix.keys():
            # Itera sobre las claves de la matriz de pagos. Cada clave es una tupla (estrategia del jugador 1, estrategia del jugador 2).
            strategies.add(s1)  # Agregar estrategia del jugador 1 al conjunto.
            strategies.add(s2)  # Agregar estrategia del jugador 2 al conjunto.
            # Al agregar al conjunto, las estrategias duplicadas se ignoran automáticamente.
        return sorted(list(strategies))  # Devolver estrategias únicas como una lista ordenada.
        # Convierte el conjunto de estrategias únicas a una lista y luego la ordena alfabéticamente antes de devolverla.

    def find_nash_equilibria(self):
        """
        Encuentra todos los equilibrios de Nash usando búsqueda en grafos.
        En este contexto, la búsqueda en grafos se implementa utilizando BFS para explorar todas las combinaciones de estrategias
        y verificar si cada combinación constituye un equilibrio de Nash.

        Returns:
            list: Lista de tuplas representando estrategias que son equilibrios de Nash.
                Cada tupla contiene la estrategia del jugador 1 y la estrategia del jugador 2 que forman un equilibrio.
        """
        equilibria = []  # Lista para almacenar los equilibrios de Nash encontrados.

        # Usamos una cola para realizar búsqueda en anchura (BFS).
        queue = deque()
        # Inicializa una cola (deque) para la búsqueda en anchura.
        queue.append((None, None))  # Nodo inicial (sin estrategias definidas aún).
        # Agrega un nodo inicial a la cola. Este nodo se utiliza para comenzar la exploración de todas las combinaciones de estrategias.

        # Conjunto para rastrear combinaciones ya visitadas.
        visited = set()
        # Inicializa un conjunto vacío llamado 'visited' para almacenar las combinaciones de estrategias que ya han sido examinadas.
        # Esto evita ciclos y redundancia en la búsqueda.

        while queue:
            # Continúa mientras la cola no esté vacía, lo que significa que aún hay combinaciones de estrategias para explorar.
            current = queue.popleft()  # Extraer el siguiente nodo (combinación de estrategias) de la cola.

            # Si es el nodo inicial, generar todas las combinaciones posibles de estrategias.
            if current == (None, None):
                # Si el nodo extraído es el nodo inicial (sin estrategias), esto indica el comienzo de la búsqueda.
                for s1 in self.strategies:
                    # Itera sobre todas las estrategias posibles para el jugador 1.
                    for s2 in self.strategies:
                        # Itera sobre todas las estrategias posibles para el jugador 2.
                        combination = (s1, s2)
                        # Crea una tupla que representa una combinación de estrategias.
                        if combination not in visited:
                            # Si esta combinación de estrategias aún no ha sido visitada.
                            visited.add(combination)  # Marcar la combinación como visitada.
                            queue.append(combination)  # Agregar la combinación a la cola para su posterior evaluación.
                continue  # Pasar al siguiente ciclo del bucle while sin realizar la verificación de equilibrio en el nodo inicial.

            s1, s2 = current  # Estrategias actuales de los jugadores extraídas del nodo de la cola.

            # Verificar si la combinación actual es un equilibrio de Nash.
            is_nash = True
            # Inicializa una variable booleana 'is_nash' como True. Se cambiará a False si se encuentra una estrategia mejor para alguno de los jugadores.

            # Verificar para el jugador 1.
            best_p1 = self.payoff_matrix[(s1, s2)][0]  # Pago actual del jugador 1 dada la combinación de estrategias actual.
            for alt_s1 in self.strategies:  # Probar todas las estrategias alternativas que el jugador 1 podría elegir.
                if self.payoff_matrix[(alt_s1, s2)][0] > best_p1:
                    # Si el pago que el jugador 1 obtendría al cambiar a la estrategia 'alt_s1' (manteniendo la estrategia del jugador 2 's2')
                    # es mayor que su pago actual ('best_p1').
                    is_nash = False  # La combinación actual no es un equilibrio de Nash para el jugador 1 porque tiene un incentivo para cambiar su estrategia.
                    break  # Sale del bucle de estrategias alternativas del jugador 1 ya que se ha determinado que no es un equilibrio.

            # Verificar para el jugador 2.
            if is_nash:  # Solo verificar para el jugador 2 si la condición de equilibrio ya se cumple para el jugador 1.
                best_p2 = self.payoff_matrix[(s1, s2)][1]  # Pago actual del jugador 2 dada la combinación de estrategias actual.
                for alt_s2 in self.strategies:  # Probar todas las estrategias alternativas que el jugador 2 podría elegir.
                    if self.payoff_matrix[(s1, alt_s2)][1] > best_p2:
                        # Si el pago que el jugador 2 obtendría al cambiar a la estrategia 'alt_s2' (manteniendo la estrategia del jugador 1 's1')
                        # es mayor que su pago actual ('best_p2').
                        is_nash = False  # La combinación actual no es un equilibrio de Nash para el jugador 2 porque tiene un incentivo para cambiar su estrategia.
                        break  # Sale del bucle de estrategias alternativas del jugador 2.

            # Si cumple las condiciones para ambos jugadores, es un equilibrio de Nash.
            if is_nash:
                # Si después de verificar todas las estrategias alternativas para ambos jugadores, ninguno tiene un incentivo para cambiar,
                # entonces la combinación actual de estrategias es un equilibrio de Nash.
                equilibria.append((s1, s2))
                # Agrega la tupla de estrategias (s1, s2) a la lista de equilibrios de Nash encontrados.

        return equilibria  # Devolver la lista de todos los equilibrios de Nash encontrados.


# Ejemplo: Juego de Halcones y Palomas (Hawk-Dove Game)
def hawk_dove_example():
    """
    Ejemplo del juego Halcón-Paloma.
    - Halcón (H): Estrategia agresiva.
    - Paloma (D): Estrategia pacífica.

    Matriz de pagos:
          H       D
    H   (-2,-2) (4,0)
    D   (0,4)   (2,2)

    Interpretación:
    - Cuando dos halcones se encuentran: pelean (pago negativo).
    - Halcón vs Paloma: Halcón gana el recurso.
    - Dos palomas: comparten el recurso.
    """
    # Definir la matriz de pagos del juego.
    payoff_matrix = {
        ('H', 'H'): (-2, -2),  # Ambos jugadores eligen Halcón (conflicto).
        ('H', 'D'): (4, 0),   # Jugador 1 elige Halcón, jugador 2 elige Paloma.
        ('D', 'H'): (0, 4),   # Jugador 1 elige Paloma, jugador 2 elige Halcón.
        ('D', 'D'): (2, 2)    # Ambos jugadores eligen Paloma (cooperación).
    }

    # Crear el grafo del juego.
    game = GameGraph(payoff_matrix)
    # Crea una instancia de la clase GameGraph con la matriz de pagos definida para el juego Halcón-Paloma.

    # Encontrar los equilibrios de Nash.
    equilibria = game.find_nash_equilibria()
    # Llama al método find_nash_equilibria de la instancia de GameGraph para encontrar todos los equilibrios de Nash en el juego.

    # Mostrar la matriz de pagos.
    print("Matriz de pagos del juego Halcón-Paloma:")
    print("     H       D")
    print(f"H   {payoff_matrix[('H', 'H')]} {payoff_matrix[('H', 'D')]}")  # Fila H.
    print(f"D   {payoff_matrix[('D', 'H')]} {payoff_matrix[('D', 'D')]}")  # Fila D.

    # Mostrar los equilibrios de Nash encontrados.
    print("\nEquilibrios de Nash encontrados:")
    for eq in equilibria:
        # Itera sobre la lista de equilibrios de Nash encontrados.
        print(f"- {eq}")
        # Imprime cada equilibrio de Nash, que es una tupla de estrategias (estrategia del jugador 1, estrategia del jugador 2).

    # Interpretar los resultados.
    print("\nInterpretación:")
    if ('D', 'H') in equilibria and ('H', 'D') in equilibria:
        # Verifica si los equilibrios donde un jugador es Halcón y el otro es Paloma están presentes.
        print("Existen dos equilibrios de Nash en estrategias puras:")
        print("1. Un jugador es Halcón y el otro es Paloma.")
        print("2. Viceversa.")
    elif ('D', 'D') in equilibria:
        # Verifica si el equilibrio donde ambos jugadores son Paloma está presente.
        print("El equilibrio es que ambos jueguen Paloma (cooperación).")
    elif ('H', 'H') in equilibria:
        # Verifica si el equilibrio donde ambos jugadores son Halcón está presente.
        print("El equilibrio es que ambos jueguen Halcón (conflicto).")

# Punto de entrada principal.
if __name__ == "__main__":
    # Este bloque de código se ejecuta solo si el script se llama directamente (no cuando se importa como un módulo).
    hawk_dove_example()
    # Llama a la función hawk_dove_example para ejecutar la simulación del juego Halcón-Paloma y encontrar sus equilibrios de Nash.