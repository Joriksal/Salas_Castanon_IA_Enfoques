import numpy as np
from collections import defaultdict

class POMDP:
    def __init__(self, estados, acciones, observaciones, transiciones, observacion_probs, recompensas, gamma=0.95):
        """
        Inicializa un POMDP (Proceso de Decisión de Markov Parcialmente Observable).
        
        Args:
            estados: Lista de estados posibles.
            acciones: Lista de acciones posibles.
            observaciones: Lista de observaciones posibles.
            transiciones: Diccionario {(estado, accion): {estado_siguiente: probabilidad}}.
            observacion_probs: Diccionario {(accion, estado_siguiente): {observacion: probabilidad}}.
            recompensas: Diccionario {(estado, accion): recompensa}.
            gamma: Factor de descuento para recompensas futuras.
        """
        self.estados = estados
        self.acciones = acciones
        self.observaciones = observaciones
        self.transiciones = transiciones
        self.observacion_probs = observacion_probs
        self.recompensas = recompensas
        self.gamma = gamma
        # Inicializar creencias uniformes (probabilidad igual para todos los estados).
        self.credencias = np.ones(len(estados)) / len(estados)

    def actualizar_credencias(self, accion, observacion):
        """
        Actualiza las creencias del agente usando el filtro de Bayes.
        
        Args:
            accion: Acción tomada.
            observacion: Observación recibida.
            
        Returns:
            np.array: Nuevo vector de creencias.
        """
        nuevas_credencias = np.zeros(len(self.estados))
        
        # Iterar sobre todos los estados posibles (s').
        for i, s_prime in enumerate(self.estados):
            prob = 0  # Probabilidad acumulada para el estado s'.
            for j, s in enumerate(self.estados):
                # P(s'|s,a) * P(o|a,s') * b(s)
                prob += self.transiciones.get((s, accion), {}).get(s_prime, 0) * \
                        self.observacion_probs.get((accion, s_prime), {}).get(observacion, 0) * \
                        self.credencias[j]
            nuevas_credencias[i] = prob
        
        # Normalizar las creencias para que sumen 1.
        if np.sum(nuevas_credencias) > 0:
            nuevas_credencias /= np.sum(nuevas_credencias)
        else:
            # Si no hay información, volver a una creencia uniforme.
            nuevas_credencias = np.ones(len(self.estados)) / len(self.estados)
        
        self.credencias = nuevas_credencias
        return nuevas_credencias

    def paso_pomdp(self, accion):
        """
        Realiza un paso en el POMDP: ejecuta acción, obtiene observación y recompensa.
        
        Args:
            accion: Acción a ejecutar.
            
        Returns:
            tuple: (observacion, recompensa, nueva_creencia).
        """
        # 1. Seleccionar el estado real basado en las creencias actuales.
        estado_real = np.random.choice(self.estados, p=self.credencias)
        
        # 2. Determinar el nuevo estado basado en las transiciones.
        trans_dict = self.transiciones.get((estado_real, accion), {})
        estados_posibles = list(trans_dict.keys())
        probs = list(trans_dict.values())
        
        if not estados_posibles:
            # Si no hay transiciones definidas, permanecer en el mismo estado.
            nuevo_estado = estado_real
        else:
            # Elegir el nuevo estado basado en las probabilidades de transición.
            nuevo_estado = np.random.choice(estados_posibles, p=probs)
        
        # 3. Generar una observación basada en el nuevo estado.
        obs_dict = self.observacion_probs.get((accion, nuevo_estado), {})
        observaciones_posibles = list(obs_dict.keys())
        obs_probs = list(obs_dict.values())
        
        if not observaciones_posibles:
            # Si no hay observaciones definidas, devolver None.
            observacion = None
        else:
            # Elegir una observación basada en las probabilidades.
            observacion = np.random.choice(observaciones_posibles, p=obs_probs)
        
        # 4. Obtener la recompensa asociada a la acción tomada.
        recompensa = self.recompensas.get((estado_real, accion), 0)
        
        # 5. Actualizar las creencias del agente.
        nuevas_credencias = self.actualizar_credencias(accion, observacion)
        
        return observacion, recompensa, nuevas_credencias

    def resolver_punto_vista(self, horizonte=10, n_muestras=1000):
        """
        Resuelve el POMDP mediante simulaciones para encontrar la mejor acción.
        
        Args:
            horizonte: Número de pasos a futuro a considerar.
            n_muestras: Número de simulaciones por acción.
            
        Returns:
            str: Mejor acción a tomar desde las creencias actuales.
        """
        mejor_accion = None
        mejor_valor = -np.inf
        
        # Evaluar cada acción posible.
        for accion in self.acciones:
            valor_total = 0
            
            # Realizar simulaciones para esta acción.
            for _ in range(n_muestras):
                creencias = self.credencias.copy()
                valor_trayectoria = 0
                
                # Simular una trayectoria de longitud "horizonte".
                for t in range(horizonte):
                    # 1. Ejecutar la acción inicial o una acción aleatoria después.
                    a = accion if t == 0 else np.random.choice(self.acciones)
                    
                    # 2. Simular el estado real basado en las creencias.
                    estado_real = np.random.choice(self.estados, p=creencias)
                    trans_dict = self.transiciones.get((estado_real, a), {})
                    estados_posibles = list(trans_dict.keys())
                    probs = list(trans_dict.values())
                    
                    if not estados_posibles:
                        nuevo_estado = estado_real
                    else:
                        nuevo_estado = np.random.choice(estados_posibles, p=probs)
                    
                    # 3. Obtener la recompensa para esta acción.
                    valor_trayectoria += (self.gamma**t) * self.recompensas.get((estado_real, a), 0)
                    
                    # 4. Simular observación y actualizar creencias.
                    obs_dict = self.observacion_probs.get((a, nuevo_estado), {})
                    observaciones_posibles = list(obs_dict.keys())
                    obs_probs = list(obs_dict.values())
                    
                    if observaciones_posibles:
                        observacion = np.random.choice(observaciones_posibles, p=obs_probs)
                        nuevas_credencias = np.zeros(len(self.estados))
                        
                        for i, s_prime in enumerate(self.estados):
                            prob = 0
                            for j, s in enumerate(self.estados):
                                prob += self.transiciones.get((s, a), {}).get(s_prime, 0) * \
                                        obs_dict.get(observacion, 0) * \
                                        creencias[j]
                            nuevas_credencias[i] = prob
                        
                        if np.sum(nuevas_credencias) > 0:
                            nuevas_credencias /= np.sum(nuevas_credencias)
                            creencias = nuevas_credencias
                
                valor_total += valor_trayectoria
            
            # Calcular el valor promedio de la acción.
            valor_promedio = valor_total / n_muestras
            
            # Actualizar la mejor acción si el valor promedio es mayor.
            if valor_promedio > mejor_valor:
                mejor_valor = valor_promedio
                mejor_accion = accion
        
        return mejor_accion

# --------------------------------------------
# Ejemplo: Problema del Robot en Laberinto POMDP
# --------------------------------------------
def ejemplo_robot_pomdp():
    """
    Simula un robot en un laberinto usando un POMDP.
    """
    # Estados: Posición en laberinto 2x2.
    estados = ['A', 'B', 'C', 'D']
    
    # Acciones: Movimientos posibles.
    acciones = ['arriba', 'abajo', 'izquierda', 'derecha']
    
    # Observaciones: Señales percibidas por el robot.
    observaciones = ['pared_izq', 'pared_der', 'nada']
    
    # Transiciones (determinísticas en este ejemplo).
    transiciones = {
        ('A', 'derecha'): {'B': 1.0},
        ('A', 'abajo'): {'C': 1.0},
        ('B', 'izquierda'): {'A': 1.0},
        ('B', 'abajo'): {'D': 1.0},
        ('C', 'arriba'): {'A': 1.0},
        ('C', 'derecha'): {'D': 1.0},
        ('D', 'izquierda'): {'C': 1.0},
        ('D', 'arriba'): {'B': 1.0},
    }
    
    # Probabilidades de observación (ruidosas).
    observacion_probs = {
        ('arriba', 'A'): {'pared_izq': 0.8, 'nada': 0.2},
        ('arriba', 'B'): {'pared_der': 0.8, 'nada': 0.2},
        ('arriba', 'C'): {'nada': 1.0},
        ('arriba', 'D'): {'nada': 1.0},
    }
    
    # Recompensas: Meta en estado D.
    recompensas = {
        ('D', 'arriba'): 10,
        ('D', 'izquierda'): 10,
    }
    
    # Crear el POMDP.
    robot_pomdp = POMDP(estados, acciones, observaciones, transiciones, 
                        observacion_probs, recompensas, gamma=0.9)
    
    # Ejecutar simulación.
    print("Simulación de POMDP - Robot en Laberinto")
    print("Estado inicial: creencia uniforme")
    
    for paso in range(5):
        # Resolver para acción óptima.
        accion = robot_pomdp.resolver_punto_vista(horizonte=3, n_muestras=500)
        
        # Ejecutar paso.
        obs, recompensa, nuevas_credencias = robot_pomdp.paso_pomdp(accion)
        
        print(f"\nPaso {paso + 1}:")
        print(f"Acción tomada: {accion}")
        print(f"Observación: {obs}")
        print(f"Recompensa: {recompensa}")
        print(f"Nuevas creencias: {dict(zip(estados, np.round(nuevas_credencias, 3)))}")
    
    return robot_pomdp

if __name__ == "__main__":
    ejemplo_robot_pomdp()