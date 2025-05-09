# Importa la librería numpy, útil para manejar arreglos, probabilidades y números aleatorios
import numpy as np

# Clase para definir y operar un modelo POMDP (Proceso de Decisión de Markov Parcialmente Observable)
class POMDP:
    def __init__(self, estados, acciones, observaciones, transiciones, observacion_probs, recompensas, gamma=0.95):
        """
        Inicializa el POMDP con sus elementos fundamentales:
        
        - estados: lista de todos los estados posibles del sistema (por ejemplo, posiciones del robot).
        - acciones: acciones que el agente puede ejecutar (por ejemplo, moverse a la derecha, izquierda...).
        - observaciones: lo que el agente puede "ver" o "percibir", aunque no vea el estado completo.
        - transiciones: diccionario que define la probabilidad de ir de un estado a otro con cierta acción.
        - observacion_probs: diccionario que define la probabilidad de observar algo dado un estado y acción.
        - recompensas: define cuánto gana (o pierde) el agente por hacer ciertas acciones en ciertos estados.
        - gamma: factor de descuento (entre 0 y 1) que da menos valor a recompensas futuras.
        """
        self.estados = estados
        self.acciones = acciones
        self.observaciones = observaciones
        self.transiciones = transiciones
        self.observacion_probs = observacion_probs
        self.recompensas = recompensas
        self.gamma = gamma
        self.credencias = np.ones(len(estados)) / len(estados)  # Distribución de creencias uniforme al inicio

    def actualizar_credencias(self, accion, observacion):
        """
        Actualiza la creencia del agente sobre en qué estado se encuentra, después de hacer una acción
        y recibir una observación. Usa el teorema de Bayes.
        """
        nuevas_credencias = np.zeros(len(self.estados))  # Se arma un nuevo vector para las nuevas creencias

        for i, s_prime in enumerate(self.estados):  # s_prime: posibles estados actuales tras la acción
            prob = 0
            for j, s in enumerate(self.estados):  # s: posibles estados anteriores
                # Probabilidad de transitar de s a s_prime con la acción dada
                trans = self.transiciones.get((s, accion), {}).get(s_prime, 0)
                # Probabilidad de haber observado lo que se observó si ahora se está en s_prime
                obs_prob = self.observacion_probs.get((accion, s_prime), {}).get(observacion, 0)
                # Suma ponderada con la creencia previa
                prob += trans * obs_prob * self.credencias[j]
            nuevas_credencias[i] = prob  # Asigna la nueva creencia de estar en s_prime

        # Normaliza el vector de creencias para que la suma sea 1 (probabilidad válida)
        if np.sum(nuevas_credencias) > 0:
            nuevas_credencias /= np.sum(nuevas_credencias)
        else:
            # Si todo da cero (por ejemplo, observación imposible), se reinicia con probabilidad uniforme
            nuevas_credencias = np.ones(len(self.estados)) / len(self.estados)

        self.credencias = nuevas_credencias  # Actualiza la creencia interna del modelo
        return nuevas_credencias

    def paso_pomdp(self, accion):
        """
        Ejecuta un paso en el entorno POMDP:

        - Escoge un estado real basado en las creencias.
        - Se ejecuta la acción y se transita a un nuevo estado.
        - Se genera una observación basada en ese nuevo estado.
        - Se obtiene la recompensa correspondiente.
        - Se actualizan las creencias según la observación.
        """
        # Escoge un estado real aleatorio basado en las creencias actuales
        estado_real = np.random.choice(self.estados, p=self.credencias)

        # Diccionario con probabilidades de transición desde ese estado y acción: {(estado_real, accion): {s': p}}
        trans_dict = self.transiciones.get((estado_real, accion), {})
        estados_posibles = list(trans_dict.keys())
        probs = list(trans_dict.values())

        # Se elige un nuevo estado al que se transita según esas probabilidades
        nuevo_estado = np.random.choice(estados_posibles, p=probs) if estados_posibles else estado_real

        # ---------- AQUÍ VA TU DUDA: ¿QUÉ HACE obs_dict? ----------
        # Este diccionario contiene: qué observaciones puede recibir el agente al estar en 'nuevo_estado' después de aplicar 'accion'.
        # Formato: {observación1: prob1, observación2: prob2, ...}
        obs_dict = self.observacion_probs.get((accion, nuevo_estado), {})
        observaciones_posibles = list(obs_dict.keys())  # Observaciones que podrían recibirse
        obs_probs = list(obs_dict.values())            # Sus respectivas probabilidades

        # Se elige aleatoriamente una observación según las probabilidades
        observacion = np.random.choice(observaciones_posibles, p=obs_probs) if observaciones_posibles else None

        # Obtiene la recompensa por ejecutar la acción en el estado real
        recompensa = self.recompensas.get((estado_real, accion), 0)

        # Se actualizan las creencias del agente con base en la observación recibida
        nuevas_credencias = self.actualizar_credencias(accion, observacion)

        return observacion, recompensa, nuevas_credencias

    def resolver_punto_vista(self, horizonte=10, n_muestras=1000):
        """
        Encuentra la mejor acción posible al simular muchas trayectorias.
        
        - horizonte: cuántos pasos futuros se considera en la simulación.
        - n_muestras: cuántas simulaciones se hacen por acción.
        """
        mejor_accion = None
        mejor_valor = -np.inf

        for accion in self.acciones:  # Se evalúa cada acción posible
            valor_total = 0

            for _ in range(n_muestras):  # Se hacen n simulaciones para esa acción
                creencias = self.credencias.copy()  # Se copia la creencia actual para no modificar la real
                valor_trayectoria = 0

                for t in range(horizonte):  # Para cada paso dentro del horizonte
                    a = accion if t == 0 else np.random.choice(self.acciones)

                    # Escoge estado real con base en creencias
                    estado_real = np.random.choice(self.estados, p=creencias)

                    # Se hace la transición
                    trans_dict = self.transiciones.get((estado_real, a), {})
                    estados_posibles = list(trans_dict.keys())
                    probs = list(trans_dict.values())
                    nuevo_estado = np.random.choice(estados_posibles, p=probs) if estados_posibles else estado_real

                    # Acumula recompensa con descuento
                    valor_trayectoria += (self.gamma**t) * self.recompensas.get((estado_real, a), 0)

                    # Obtiene observaciones posibles desde el nuevo estado
                    obs_dict = self.observacion_probs.get((a, nuevo_estado), {})  # <<------ Aquí también se usa obs_dict
                    observaciones_posibles = list(obs_dict.keys())
                    obs_probs = list(obs_dict.values())

                    # Genera observación
                    if observaciones_posibles:
                        observacion = np.random.choice(observaciones_posibles, p=obs_probs)

                        # Actualiza creencias
                        nuevas_credencias = np.zeros(len(self.estados))
                        for i, s_prime in enumerate(self.estados):
                            prob = 0
                            for j, s in enumerate(self.estados):
                                trans = self.transiciones.get((s, a), {}).get(s_prime, 0)
                                prob += trans * obs_dict.get(observacion, 0) * creencias[j]
                            nuevas_credencias[i] = prob

                        # Normaliza creencias si hay probabilidad válida
                        if np.sum(nuevas_credencias) > 0:
                            nuevas_credencias /= np.sum(nuevas_credencias)
                            creencias = nuevas_credencias

                valor_total += valor_trayectoria  # Suma valor total de esa simulación

            # Calcula el promedio del valor de todas las simulaciones para esa acción
            valor_promedio = valor_total / n_muestras

            # Si es mejor que el valor actual, la guardamos como mejor acción
            if valor_promedio > mejor_valor:
                mejor_valor = valor_promedio
                mejor_accion = accion

        return mejor_accion  # Devuelve la acción más prometedora

# ------------------------------------------------------------------
# Función para probar el POMDP con un ejemplo de robot en un laberinto
def ejemplo_robot_pomdp():
    """
    Ejemplo sencillo: un robot en un laberinto 2x2.
    Estados: A, B, C, D (posiciones en la cuadrícula).
    Acciones: moverse en 4 direcciones.
    Observaciones: si hay pared a la izquierda, derecha o nada.
    """
    estados = ['A', 'B', 'C', 'D']
    acciones = ['arriba', 'abajo', 'izquierda', 'derecha']
    observaciones = ['pared_izq', 'pared_der', 'nada']

    # Transiciones entre celdas del laberinto (completamente determinísticas en este ejemplo)
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

    # Probabilidad de observaciones dadas ciertas acciones y estados
    observacion_probs = {
        ('arriba', 'A'): {'pared_izq': 0.8, 'nada': 0.2},
        ('arriba', 'B'): {'pared_der': 0.8, 'nada': 0.2},
        ('arriba', 'C'): {'nada': 1.0},
        ('arriba', 'D'): {'nada': 1.0},
    }

    # Recompensas por llegar a ciertos estados o tomar ciertas acciones
    recompensas = {
        ('D', 'arriba'): 10,
        ('D', 'izquierda'): 10,
    }

    # Creamos el objeto POMDP con todos los componentes
    robot_pomdp = POMDP(estados, acciones, observaciones, transiciones,
                        observacion_probs, recompensas, gamma=0.9)

    print("Simulación de POMDP - Robot en Laberinto")

    for paso in range(5):
        accion = robot_pomdp.resolver_punto_vista(horizonte=3, n_muestras=500)
        obs, recompensa, nuevas_credencias = robot_pomdp.paso_pomdp(accion)

        print(f"\nPaso {paso + 1}:")
        print(f"Acción tomada: {accion}")
        print(f"Observación recibida: {obs}")
        print(f"Recompensa obtenida: {recompensa}")
        print(f"Nuevas creencias: {dict(zip(estados, np.round(nuevas_credencias, 3)))}")

    return robot_pomdp

# Si este archivo se ejecuta directamente, corre el ejemplo
if __name__ == "__main__":
    ejemplo_robot_pomdp()
