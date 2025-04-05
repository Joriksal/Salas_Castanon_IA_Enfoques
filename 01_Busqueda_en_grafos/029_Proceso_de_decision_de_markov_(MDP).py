import numpy as np
from collections import defaultdict

class MDP:
    def __init__(self, estados, acciones, transiciones, recompensas, gamma=0.95):
        """
        Inicializa un Proceso de Decisión de Markov (MDP).
        
        Args:
            estados: Lista de estados posibles.
            acciones: Lista de acciones posibles.
            transiciones: Diccionario {(estado, accion): {estado_siguiente: probabilidad}}.
            recompensas: Diccionario {(estado, accion, estado_siguiente): recompensa}.
            gamma: Factor de descuento (default: 0.95), que pondera la importancia de recompensas futuras.
        """
        self.estados = estados
        self.acciones = acciones
        self.transiciones = transiciones
        self.recompensas = recompensas
        self.gamma = gamma  # Factor de descuento para recompensas futuras.
        self.grafo = self._construir_grafo()  # Construir representación del grafo del MDP.

    def _construir_grafo(self):
        """
        Construye una representación del grafo del MDP.
        
        Returns:
            dict: Grafo que conecta estados y acciones con sus transiciones.
        """
        grafo = defaultdict(dict)
        for (s, a), destinos in self.transiciones.items():
            for s_prime, prob in destinos.items():
                grafo[(s, a)][s_prime] = {
                    'probabilidad': prob,
                    'recompensa': self.recompensas.get((s, a, s_prime), 0)
                }
        return grafo

    def iteracion_valores(self, epsilon=1e-6, max_iter=1000):
        """
        Algoritmo de Iteración de Valores para resolver el MDP.
        
        Args:
            epsilon: Criterio de convergencia (default: 1e-6).
            max_iter: Máximo número de iteraciones (default: 1000).
            
        Returns:
            tuple: (valores óptimos, política óptima).
        """
        # Inicializar los valores de todos los estados en 0.
        V = {s: 0 for s in self.estados}
        
        for _ in range(max_iter):
            delta = 0  # Diferencia máxima entre valores antiguos y nuevos.
            V_nuevo = V.copy()  # Copia de los valores actuales para actualizarlos.
            
            # Iterar sobre todos los estados.
            for s in self.estados:
                # Si el estado no tiene transiciones, es terminal.
                if s not in [key[0] for key in self.transiciones.keys()]:
                    continue
                
                Q = {}  # Diccionario para almacenar los valores Q(s, a).
                for a in self.acciones:
                    # Si no hay transiciones definidas para esta acción, saltar.
                    if (s, a) not in self.transiciones:
                        continue
                        
                    # Calcular el valor esperado para esta acción.
                    q_val = 0
                    for s_prime, prob in self.transiciones[(s, a)].items():
                        r = self.recompensas.get((s, a, s_prime), 0)  # Recompensa inmediata.
                        q_val += prob * (r + self.gamma * V[s_prime])  # Valor esperado.
                    Q[a] = q_val  # Guardar el valor Q(s, a).
                
                # Actualizar el valor del estado con el máximo Q(s, a).
                if Q:
                    V_nuevo[s] = max(Q.values())
                    delta = max(delta, abs(V_nuevo[s] - V[s]))  # Actualizar el cambio máximo.
            
            V = V_nuevo  # Actualizar los valores de los estados.
            if delta < epsilon:  # Verificar criterio de convergencia.
                break
        
        # Extraer la política óptima.
        politica = {}
        for s in self.estados:
            # Si el estado no tiene transiciones, no hay acción óptima.
            if s not in [key[0] for key in self.transiciones.keys()]:
                politica[s] = None
                continue
                
            Q = {}  # Diccionario para almacenar los valores Q(s, a).
            for a in self.acciones:
                if (s, a) not in self.transiciones:
                    continue
                    
                # Calcular el valor esperado para esta acción.
                q_val = 0
                for s_prime, prob in self.transiciones[(s, a)].items():
                    r = self.recompensas.get((s, a, s_prime), 0)
                    q_val += prob * (r + self.gamma * V[s_prime])
                Q[a] = q_val
            
            # Seleccionar la acción con el mayor valor Q(s, a).
            politica[s] = max(Q, key=Q.get) if Q else None
        
        return V, politica

    def visualizar_grafo(self):
        """
        Visualización básica del grafo del MDP.
        """
        print("\nGrafo del MDP (Estado, Acción) -> [Destinos]:")
        for (s, a), destinos in self.grafo.items():
            print(f"\nDesde ({s}, {a}):")
            for s_prime, datos in destinos.items():
                print(f"  → {s_prime} [P={datos['probabilidad']:.2f}, R={datos['recompensa']}]")

# --------------------------------------------
# Ejemplo: Problema del Inventario
# --------------------------------------------
def ejemplo_inventario():
    """
    Define y resuelve un problema de inventario usando Iteración de Valores.
    
    Returns:
        tuple: (MDP, valores óptimos, política óptima).
    """
    # Estados: Nivel de inventario (0-2 unidades).
    estados = [0, 1, 2]
    
    # Acciones: Pedir 0, 1 o 2 unidades.
    acciones = [0, 1, 2]
    
    # Dinámica de transiciones.
    transiciones = {
        # (estado_actual, accion): {estado_siguiente: probabilidad}.
        (0, 0): {0: 0.5, 1: 0.3, 2: 0.2},
        (0, 1): {1: 0.5, 2: 0.3, 0: 0.2},
        (0, 2): {2: 0.5, 1: 0.3, 0: 0.2},
        
        (1, 0): {0: 0.6, 1: 0.3, 2: 0.1},
        (1, 1): {1: 0.6, 2: 0.3, 0: 0.1},
        (1, 2): {2: 0.6, 1: 0.3, 0: 0.1},
        
        (2, 0): {1: 0.7, 2: 0.2, 0: 0.1},
        (2, 1): {2: 0.7, 1: 0.2, 0: 0.1},
        (2, 2): {2: 0.8, 1: 0.1, 0: 0.1}
    }
    
    # Función de recompensa.
    recompensas = {
        # (estado, accion, estado_siguiente): recompensa.
        (0, 0, 0): -10,  # Costo por falta de inventario.
        (0, 1, 1): -2,   # Costo de almacenamiento.
        (0, 2, 2): -4,
        (1, 0, 0): 5,     # Beneficio por venta.
        (1, 1, 1): 3,
        (2, 0, 1): 8,
        (2, 2, 2): 2
    }
    
    # Crear el MDP.
    mdp_inventario = MDP(estados, acciones, transiciones, recompensas, gamma=0.9)
    
    # Visualizar estructura del grafo.
    mdp_inventario.visualizar_grafo()
    
    # Resolver con Iteración de Valores.
    valores, politica = mdp_inventario.iteracion_valores()
    
    # Mostrar resultados.
    print("\nValores óptimos:")
    for s in estados:
        print(f"Estado {s}: {valores[s]:.2f}")
    
    print("\nPolítica óptima:")
    for s in estados:
        print(f"En estado {s}: pedir {politica[s]} unidades")
    
    return mdp_inventario, valores, politica

if __name__ == "__main__":
    # Ejecutar el ejemplo del inventario.
    ejemplo_inventario()