# Importamos defaultdict para manejar diccionarios con valores por defecto.
from collections import defaultdict

class MDP:
    def __init__(self, estados, acciones, recompensas, transiciones, gamma=0.9):
        """
        Inicializa un Proceso de Decisión de Markov (MDP).
        
        Args:
            estados: Lista de estados posibles.
            acciones: Lista de acciones posibles.
            recompensas: Diccionario {(estado, accion, estado_siguiente): recompensa}.
            transiciones: Diccionario {(estado, accion): {estado_siguiente: probabilidad}}.
            gamma: Factor de descuento (default: 0.9), que pondera la importancia de recompensas futuras.
        """
        # Lista de estados posibles en el MDP.
        self.estados = estados
        # Lista de acciones posibles en el MDP.
        self.acciones = acciones
        # Diccionario que define las recompensas para cada transición (estado, acción, estado_siguiente).
        self.recompensas = recompensas
        # Diccionario que define las probabilidades de transición para cada (estado, acción).
        self.transiciones = transiciones
        # Factor de descuento para recompensas futuras (entre 0 y 1).
        self.gamma = gamma

    def iteracion_valores(self, epsilon=1e-6):
        """
        Implementa el algoritmo de Iteración de Valores para resolver el MDP.
        
        Args:
            epsilon: Criterio de convergencia (default: 1e-6).
            
        Returns:
            tuple: (Valores óptimos, Política óptima).
        """
        # Inicialización: Asignar valor inicial de 0 a todos los estados.
        V = {s: 0 for s in self.estados}  # Diccionario de valores de los estados.
        politica = {s: None for s in self.estados}  # Diccionario de la política óptima.

        while True:
            delta = 0  # Diferencia máxima entre valores antiguos y nuevos.
            V_nuevo = V.copy()  # Copia de los valores actuales para actualizarlos.

            # Iterar sobre todos los estados.
            for s in self.estados:
                # Si el estado no tiene transiciones, es terminal y se omite.
                if s not in self.transiciones:
                    continue

                # Diccionario para almacenar los Q-valores de cada acción.
                Q = {}
                for a in self.acciones:
                    # Si la acción no tiene transiciones definidas, se omite.
                    if (s, a) not in self.transiciones:
                        continue

                    # Calcular el valor esperado para esta acción.
                    valor_accion = 0
                    for s_siguiente, prob in self.transiciones[(s, a)].items():
                        # Obtener la recompensa asociada a la transición.
                        recompensa = self.recompensas.get((s, a, s_siguiente), 0)
                        # Sumar el valor esperado ponderado por la probabilidad.
                        valor_accion += prob * (recompensa + self.gamma * V[s_siguiente])

                    # Guardar el valor esperado de la acción en el diccionario Q.
                    Q[a] = valor_accion

                # Si hay acciones válidas, actualizar el valor del estado y la política óptima.
                if Q:
                    # Seleccionar la acción con el mayor Q-valor.
                    mejor_accion = max(Q, key=Q.get)
                    # Actualizar el valor del estado con el Q-valor de la mejor acción.
                    V_nuevo[s] = Q[mejor_accion]
                    # Actualizar la política óptima para el estado.
                    politica[s] = mejor_accion
                    # Actualizar el cambio máximo entre valores antiguos y nuevos.
                    delta = max(delta, abs(V_nuevo[s] - V[s]))

            # Actualizar los valores de los estados.
            V = V_nuevo

            # Si el cambio máximo es menor que epsilon, detener el algoritmo.
            if delta < epsilon:
                break

        # Retornar los valores y la política óptima.
        return V, politica

# --------------------------------------------
# Ejemplo: Problema del Laberinto
# --------------------------------------------
def ejemplo_laberinto():
    """
    Define y resuelve un problema de laberinto usando Iteración de Valores.
    
    Returns:
        tuple: (MDP, valores óptimos, política óptima).
    """
    # Definir estados (posiciones en el laberinto).
    estados = [(0, 0), (0, 1), (1, 0), (1, 1), (1, 2), (2, 2)]

    # Definir acciones posibles.
    acciones = ['arriba', 'abajo', 'izquierda', 'derecha']

    # Definir recompensas.
    recompensas = {
        ((1, 2), 'derecha', (2, 2)): 10,  # Premio por llegar a la meta.
        ((0, 1), 'abajo', (0, 0)): -10,  # Penalización por caer en un hoyo.
    }

    # Definir dinámica del ambiente (transiciones).
    transiciones = defaultdict(dict)

    # Movimientos válidos en el laberinto (3x3).
    # (0,0) es un hoyo, (2,2) es la meta.
    movimientos = {
        'arriba': (-1, 0),
        'abajo': (1, 0),
        'izquierda': (0, -1),
        'derecha': (0, 1)
    }

    # Construir las transiciones para cada estado y acción.
    for s in estados:
        for a in acciones:
            dx, dy = movimientos[a]
            s_siguiente = (s[0] + dx, s[1] + dy)

            # Si el movimiento es válido (dentro del laberinto).
            if s_siguiente in estados:
                transiciones[(s, a)][s_siguiente] = 1.0  # Probabilidad 1.0 de moverse.
            else:
                # Si choca con una pared, se queda en el mismo estado.
                transiciones[(s, a)][s] = 1.0

    # Crear el MDP.
    laberinto = MDP(estados, acciones, recompensas, transiciones, gamma=0.9)

    # Resolver el MDP con Iteración de Valores.
    valores, politica = laberinto.iteracion_valores()

    # Mostrar resultados.
    print("\nValores óptimos:")
    for s in estados:
        print(f"Estado {s}: {valores[s]:.2f}")

    print("\nPolítica óptima:")
    for s in estados:
        print(f"En {s}: {politica[s]}")

    return laberinto, valores, politica

if __name__ == "__main__":
    # Ejecutar el ejemplo del laberinto.
    ejemplo_laberinto()