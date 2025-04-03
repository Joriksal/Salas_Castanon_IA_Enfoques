import numpy as np
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
        self.estados = estados
        self.acciones = acciones
        self.recompensas = recompensas
        self.transiciones = transiciones
        self.gamma = gamma  # Factor de descuento para recompensas futuras.
        
    def iteracion_valores(self, epsilon=1e-6):
        """
        Implementa el algoritmo de Iteración de Valores para resolver el MDP.
        
        Args:
            epsilon: Criterio de convergencia (default: 1e-6).
            
        Returns:
            tuple: (Valores óptimos, Política óptima).
        """
        # 1. Inicialización: Asignar valor inicial de 0 a todos los estados.
        V = {s: 0 for s in self.estados}  # Valores iniciales de los estados.
        politica = {s: None for s in self.estados}  # Política inicial (sin acciones asignadas).
        
        while True:
            delta = 0  # Diferencia máxima entre valores antiguos y nuevos.
            V_nuevo = V.copy()  # Copia de los valores actuales para actualizarlos.
            
            # 2. Iterar sobre todos los estados.
            for s in self.estados:
                if s not in self.transiciones:
                    continue  # Saltar estados terminales (sin transiciones).
                    
                # 3. Calcular el Q-valor para cada acción posible.
                Q = {}
                for a in self.acciones:
                    if (s, a) not in self.transiciones:
                        continue  # Saltar acciones no válidas.
                        
                    # 4. Calcular el valor esperado para esta acción.
                    valor_accion = 0
                    for s_siguiente, prob in self.transiciones[(s, a)].items():
                        recompensa = self.recompensas.get((s, a, s_siguiente), 0)
                        valor_accion += prob * (recompensa + self.gamma * V[s_siguiente])
                    
                    Q[a] = valor_accion  # Guardar el valor esperado de la acción.
                
                # 5. Actualizar el valor del estado y la política óptima.
                if Q:
                    mejor_accion = max(Q, key=Q.get)  # Acción con el mayor Q-valor.
                    V_nuevo[s] = Q[mejor_accion]  # Actualizar el valor del estado.
                    politica[s] = mejor_accion  # Actualizar la política óptima.
                    delta = max(delta, abs(V_nuevo[s] - V[s]))  # Actualizar el cambio máximo.
            
            V = V_nuevo  # Actualizar los valores de los estados.
            
            # 6. Criterio de convergencia: Si el cambio máximo es menor que epsilon, detener.
            if delta < epsilon:
                break
                
        return V, politica  # Retornar los valores y la política óptima.

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