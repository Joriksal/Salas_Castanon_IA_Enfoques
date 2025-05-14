# Definición de la clase que representa un agente basado en creencias
class BeliefAgent:
    def __init__(self):
        """
        Constructor de la clase BeliefAgent.
        Inicializa el estado real del mundo y las creencias del agente.
        """
        # Estado real del mundo (este estado es oculto para el agente, 
        # ya que el agente no puede acceder directamente a esta información).
        self.real_world = {
            'caja_pos': 'A',  # La posición real de la caja es 'A'.
            'obstaculo': False  # Indica que no hay un obstáculo real en la posición 'A'.
        }
        
        # Creencias del agente (estas pueden ser incorrectas o incompletas,
        # ya que dependen de lo que el agente percibe del entorno).
        self.beliefs = {
            'caja_pos': 'A',  # El agente cree que la caja está en 'A'.
            'obstaculo': False  # El agente cree que no hay un obstáculo.
        }

    def update_beliefs(self, event):
        """
        Actualiza las creencias del agente basándose en un evento percibido.
        Los eventos representan información que el agente recibe del entorno.

        Parámetros:
        - event (str): El evento percibido por el agente. Puede ser "ver_obstaculo" o "no_ver_obstaculo".
        """
        if event == "ver_obstaculo":
            # Si el agente percibe un obstáculo, actualiza su creencia para reflejarlo.
            self.beliefs['obstaculo'] = True
            print("El agente ahora CREE que hay un obstáculo.")
        
        elif event == "no_ver_obstaculo":
            # Si el agente no percibe un obstáculo, actualiza su creencia para reflejarlo.
            self.beliefs['obstaculo'] = False
            print("El agente ahora CREE que NO hay un obstáculo.")

    def act(self, action):
        """
        Toma una decisión basada en las creencias del agente, no en el estado real del mundo.
        Esto puede llevar a errores si las creencias no coinciden con la realidad.

        Parámetros:
        - action (str): La acción que el agente intenta realizar. En este caso, "mover_caja".
        """
        if action == "mover_caja":
            # El agente decide mover la caja solo si cree que no hay un obstáculo.
            if not self.beliefs['obstaculo']:
                print("El agente DECIDE mover la caja (sin saber si hay obstáculo real).")
                
                # Aquí se verifica si hay un obstáculo real en el mundo.
                # Nota: El agente no tiene acceso directo a esta información.
                if self.real_world['obstaculo']:
                    # Si hay un obstáculo real, la acción falla.
                    print("¡Fallo! Había un obstáculo real.")
                else:
                    # Si no hay un obstáculo real, la acción tiene éxito.
                    print("¡Éxito! No había obstáculo.")
            else:
                # Si el agente cree que hay un obstáculo, no intenta mover la caja.
                print("El agente NO mueve la caja (cree que hay obstáculo).")

# --- Ejemplo de uso ---
if __name__ == "__main__":
    """
    Punto de entrada del programa.
    Aquí se crea una instancia del agente y se simulan eventos y acciones
    para demostrar cómo el agente actualiza sus creencias y toma decisiones.
    """
    # Crear una instancia del agente
    robot = BeliefAgent()
    
    # Evento 1: El robot percibe un obstáculo (actualiza sus creencias).
    # Esto simula que el agente recibe información del entorno indicando que hay un obstáculo.
    robot.update_beliefs("ver_obstaculo")
    
    # Acción 1: El robot intenta mover la caja (basado en sus creencias).
    # En este caso, no actuará porque cree que hay un obstáculo.
    robot.act("mover_caja")
    
    # Evento 2: El robot percibe que no hay obstáculo (actualiza sus creencias).
    # Esto simula que el agente recibe información del entorno indicando que ya no hay un obstáculo.
    robot.update_beliefs("no_ver_obstaculo")
    
    # Acción 2: El robot intenta mover la caja nuevamente.
    # Puede fallar si el estado real del mundo es distinto a lo que el agente cree.
    robot.act("mover_caja")