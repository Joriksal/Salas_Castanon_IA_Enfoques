import numpy as np
from collections import deque

class GameGraph:
    def __init__(self, payoff_matrix):
        """
        Inicializa el grafo del juego con una matriz de pagos.
        
        Args:
            payoff_matrix (dict): Matriz de pagos para cada jugador.
                Formato: {(estrategia_j1, estrategia_j2): (pago_j1, pago_j2)}
        """
        self.payoff_matrix = payoff_matrix
        self.strategies = self._get_unique_strategies()  # Extraer estrategias únicas del juego.
        
    def _get_unique_strategies(self):
        """
        Extrae las estrategias únicas de la matriz de pagos.
        
        Returns:
            list: Lista de estrategias únicas.
        """
        strategies = set()
        for s1, s2 in self.payoff_matrix.keys():
            strategies.add(s1)  # Agregar estrategia del jugador 1.
            strategies.add(s2)  # Agregar estrategia del jugador 2.
        return sorted(list(strategies))  # Devolver estrategias ordenadas.
    
    def find_nash_equilibria(self):
        """
        Encuentra todos los equilibrios de Nash usando búsqueda en grafos.
        
        Returns:
            list: Lista de tuplas representando estrategias que son equilibrios de Nash.
        """
        equilibria = []  # Lista para almacenar los equilibrios de Nash.
        
        # Usamos una cola para realizar búsqueda en anchura (BFS).
        queue = deque()
        queue.append((None, None))  # Nodo inicial (sin estrategias definidas aún).
        
        # Conjunto para rastrear combinaciones ya visitadas.
        visited = set()
        
        while queue:
            current = queue.popleft()  # Extraer el siguiente nodo de la cola.
            
            # Si es el nodo inicial, generar todas las combinaciones posibles de estrategias.
            if current == (None, None):
                for s1 in self.strategies:
                    for s2 in self.strategies:
                        combination = (s1, s2)
                        if combination not in visited:
                            visited.add(combination)  # Marcar como visitado.
                            queue.append(combination)  # Agregar a la cola.
                continue  # Pasar al siguiente nodo.
                
            s1, s2 = current  # Estrategias actuales de los jugadores.
            
            # Verificar si la combinación actual es un equilibrio de Nash.
            is_nash = True
            
            # Verificar para el jugador 1.
            best_p1 = self.payoff_matrix[(s1, s2)][0]  # Pago actual del jugador 1.
            for alt_s1 in self.strategies:  # Probar todas las estrategias alternativas.
                if self.payoff_matrix[(alt_s1, s2)][0] > best_p1:
                    is_nash = False  # No es equilibrio si hay una mejor estrategia.
                    break
            
            # Verificar para el jugador 2.
            if is_nash:  # Solo verificar si el jugador 1 ya cumple.
                best_p2 = self.payoff_matrix[(s1, s2)][1]  # Pago actual del jugador 2.
                for alt_s2 in self.strategies:  # Probar todas las estrategias alternativas.
                    if self.payoff_matrix[(s1, alt_s2)][1] > best_p2:
                        is_nash = False  # No es equilibrio si hay una mejor estrategia.
                        break
            
            # Si cumple las condiciones para ambos jugadores, es un equilibrio de Nash.
            if is_nash:
                equilibria.append((s1, s2))
        
        return equilibria  # Devolver la lista de equilibrios de Nash.


# Ejemplo: Juego de Halcones y Palomas (Hawk-Dove Game)
def hawk_dove_example():
    """
    Ejemplo del juego Halcón-Paloma.
    - Halcón (H): Estrategia agresiva.
    - Paloma (D): Estrategia pacífica.
    
    Matriz de pagos:
            H       D
        H  (-2,-2) (4,0)
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
    
    # Encontrar los equilibrios de Nash.
    equilibria = game.find_nash_equilibria()
    
    # Mostrar la matriz de pagos.
    print("Matriz de pagos del juego Halcón-Paloma:")
    print("       H       D")
    print(f"H  {payoff_matrix[('H', 'H')]} {payoff_matrix[('H', 'D')]}")  # Fila H.
    print(f"D  {payoff_matrix[('D', 'H')]} {payoff_matrix[('D', 'D')]}")  # Fila D.
    
    # Mostrar los equilibrios de Nash encontrados.
    print("\nEquilibrios de Nash encontrados:")
    for eq in equilibria:
        print(f"- {eq}")
    
    # Interpretar los resultados.
    print("\nInterpretación:")
    if ('D', 'H') in equilibria and ('H', 'D') in equilibria:
        print("Existen dos equilibrios de Nash en estrategias puras:")
        print("1. Un jugador es Halcón y el otro es Paloma.")
        print("2. Viceversa.")
    elif ('D', 'D') in equilibria:
        print("El equilibrio es que ambos jueguen Paloma (cooperación).")
    elif ('H', 'H') in equilibria:
        print("El equilibrio es que ambos jueguen Halcón (conflicto).")

# Punto de entrada principal.
if __name__ == "__main__":
    hawk_dove_example()