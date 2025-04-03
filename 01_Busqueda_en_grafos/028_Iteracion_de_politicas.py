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
    
    def evaluar_politica(self, politica, epsilon=1e-6):
        """
        Evalúa una política dada mediante iteración.
        
        Args:
            politica: Diccionario {estado: accion}.
            epsilon: Criterio de convergencia (default: 1e-6).
            
        Returns:
            Dict: Valores V(s) para cada estado.
        """
        # Inicializar los valores de todos los estados en 0.
        V = {s: 0 for s in self.estados}
        
        while True:
            delta = 0  # Diferencia máxima entre valores antiguos y nuevos.
            V_nuevo = V.copy()  # Copia de los valores actuales para actualizarlos.
            
            # Iterar sobre todos los estados.
            for s in self.estados:
                if s not in politica or politica[s] is None:
                    continue  # Saltar estados terminales o no accesibles.
                
                a = politica[s]  # Acción actual según la política.
                if (s, a) not in self.transiciones:
                    continue  # Saltar si no hay transiciones definidas para esta acción.
                
                # Calcular el nuevo valor para el estado.
                nuevo_valor = 0
                for s_siguiente, prob in self.transiciones[(s, a)].items():
                    recompensa = self.recompensas.get((s, a, s_siguiente), 0)
                    nuevo_valor += prob * (recompensa + self.gamma * V[s_siguiente])
                
                V_nuevo[s] = nuevo_valor
                delta = max(delta, abs(V_nuevo[s] - V[s]))  # Actualizar el cambio máximo.
            
            V = V_nuevo  # Actualizar los valores de los estados.
            if delta < epsilon:  # Verificar criterio de convergencia.
                break
        
        return V
    
    def mejorar_politica(self, V, politica_actual):
        """
        Mejora una política basada en los valores actuales.
        
        Args:
            V: Diccionario de valores {estado: valor}.
            politica_actual: Diccionario {estado: accion}.
            
        Returns:
            Tuple: (nueva_politica, estable).
        """
        nueva_politica = {}
        estable = True  # Indica si la política es estable (no cambia).
        
        for s in self.estados:
            if s not in self.transiciones:
                nueva_politica[s] = None  # No hay acciones posibles para este estado.
                continue
            
            # Encontrar la mejor acción para este estado.
            mejor_accion = None
            mejor_valor = -np.inf
            
            for a in self.acciones:
                if (s, a) not in self.transiciones:
                    continue  # Saltar acciones no válidas.
                
                # Calcular el valor esperado para esta acción.
                valor_accion = 0
                for s_siguiente, prob in self.transiciones[(s, a)].items():
                    recompensa = self.recompensas.get((s, a, s_siguiente), 0)
                    valor_accion += prob * (recompensa + self.gamma * V[s_siguiente])
                
                # Actualizar la mejor acción si el valor es mayor.
                if valor_accion > mejor_valor:
                    mejor_valor = valor_accion
                    mejor_accion = a
            
            nueva_politica[s] = mejor_accion
            if politica_actual.get(s) != mejor_accion:
                estable = False  # La política no es estable si cambia alguna acción.
        
        return nueva_politica, estable
    
    def iteracion_politicas(self, politica_inicial=None, max_iter=100):
        """
        Algoritmo completo de Iteración de Políticas.
        
        Args:
            politica_inicial: Política inicial (opcional).
            max_iter: Máximo de iteraciones (default: 100).
            
        Returns:
            Tuple: (politica_optima, valores_optimos).
        """
        # Inicializar política aleatoria si no se proporciona.
        if politica_inicial is None:
            politica = {s: np.random.choice(self.acciones) 
                       if s in self.transiciones else None 
                       for s in self.estados}
        else:
            politica = politica_inicial
        
        for i in range(max_iter):
            # Paso 1: Evaluación de política.
            V = self.evaluar_politica(politica)
            
            # Paso 2: Mejora de política.
            nueva_politica, estable = self.mejorar_politica(V, politica)
            
            if estable:  # Si la política no cambia, hemos convergido.
                print(f"Convergencia alcanzada en iteración {i+1}")
                break
            
            politica = nueva_politica
        else:
            print("Advertencia: Máximo de iteraciones alcanzado")
        
        return politica, V

# --------------------------------------------
# Ejemplo: Problema del Robot Limpiador
# --------------------------------------------
def ejemplo_robot_limpiador():
    """
    Define y resuelve un problema de robot limpiador usando Iteración de Políticas.
    
    Returns:
        Tuple: (MDP, política óptima, valores óptimos).
    """
    # Definir estados (habitaciones y estados de limpieza).
    estados = ['A_sucia', 'A_limpia', 'B_sucia', 'B_limpia']
    
    # Definir acciones posibles.
    acciones = ['limpiar', 'mover_A', 'mover_B', 'esperar']
    
    # Definir recompensas.
    recompensas = {
        ('A_sucia', 'limpiar', 'A_limpia'): 5,
        ('B_sucia', 'limpiar', 'B_limpia'): 5,
        ('A_sucia', 'esperar', 'A_sucia'): -1,
        ('B_sucia', 'esperar', 'B_sucia'): -1,
        ('A_limpia', 'mover_B', 'B_sucia'): 0,
        ('B_limpia', 'mover_A', 'A_sucia'): 0,
    }
    
    # Definir dinámica del ambiente (transiciones).
    transiciones = {
        ('A_sucia', 'limpiar'): {'A_limpia': 0.9, 'A_sucia': 0.1},  # 10% de fallo.
        ('A_sucia', 'mover_B'): {'B_sucia': 1.0},
        ('A_sucia', 'esperar'): {'A_sucia': 1.0},
        
        ('A_limpia', 'mover_B'): {'B_sucia': 1.0},
        ('A_limpia', 'esperar'): {'A_limpia': 1.0},
        
        ('B_sucia', 'limpiar'): {'B_limpia': 0.9, 'B_sucia': 0.1},
        ('B_sucia', 'mover_A'): {'A_sucia': 1.0},
        ('B_sucia', 'esperar'): {'B_sucia': 1.0},
        
        ('B_limpia', 'mover_A'): {'A_sucia': 1.0},
        ('B_limpia', 'esperar'): {'B_limpia': 1.0},
    }
    
    # Crear el MDP.
    robot_mdp = MDP(estados, acciones, recompensas, transiciones, gamma=0.8)
    
    # Política inicial aleatoria.
    politica_inicial = {
        'A_sucia': np.random.choice(acciones),
        'A_limpia': np.random.choice(acciones),
        'B_sucia': np.random.choice(acciones),
        'B_limpia': np.random.choice(acciones)
    }
    
    # Ejecutar Iteración de Políticas.
    politica_optima, valores = robot_mdp.iteracion_politicas(politica_inicial)
    
    # Mostrar resultados.
    print("\nPolítica óptima:")
    for estado, accion in politica_optima.items():
        print(f"{estado}: {accion}")
    
    print("\nValores óptimos:")
    for estado, valor in valores.items():
        print(f"{estado}: {valor:.2f}")
    
    return robot_mdp, politica_optima, valores

if __name__ == "__main__":
    # Ejecutar el ejemplo del robot limpiador.
    ejemplo_robot_limpiador()