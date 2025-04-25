class BeliefAgent:
    def __init__(self):
        # Estado real del mundo (oculto para el agente, no puede acceder directamente)
        self.real_world = {
            'caja_pos': 'A',  # La posición real de la caja es A
            'obstaculo': False  # No hay un obstáculo real en la posición A
        }
        
        # Creencias del agente (pueden ser incorrectas o incompletas)
        self.beliefs = {
            'caja_pos': 'A',  # El agente cree que la caja está en A
            'obstaculo': False  # El agente cree que no hay un obstáculo
        }

    def update_beliefs(self, event):
        """
        Actualiza las creencias del agente basándose en un evento percibido.
        Los eventos representan información que el agente recibe del entorno.
        """
        if event == "ver_obstaculo":
            # Si el agente percibe un obstáculo, actualiza su creencia
            self.beliefs['obstaculo'] = True
            print("El agente ahora CREE que hay un obstáculo.")
        
        elif event == "no_ver_obstaculo":
            # Si el agente no percibe un obstáculo, actualiza su creencia
            self.beliefs['obstaculo'] = False
            print("El agente ahora CREE que NO hay un obstáculo.")

    def act(self, action):
        """
        Toma una decisión basada en las creencias del agente, no en el estado real.
        Esto puede llevar a errores si las creencias no coinciden con la realidad.
        """
        if action == "mover_caja":
            # El agente decide mover la caja solo si cree que no hay un obstáculo
            if not self.beliefs['obstaculo']:
                print("El agente DECIDE mover la caja (sin saber si hay obstáculo real).")
                
                # Verifica si hay un obstáculo real en el mundo (esto no lo sabe el agente)
                if self.real_world['obstaculo']:
                    # Si hay un obstáculo real, la acción falla
                    print("¡Fallo! Había un obstáculo real.")
                else:
                    # Si no hay un obstáculo real, la acción tiene éxito
                    print("¡Éxito! No había obstáculo.")
            else:
                # Si el agente cree que hay un obstáculo, no intenta mover la caja
                print("El agente NO mueve la caja (cree que hay obstáculo).")

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Crear una instancia del agente
    robot = BeliefAgent()
    
    # Evento 1: El robot percibe un obstáculo (actualiza sus creencias)
    robot.update_beliefs("ver_obstaculo")
    
    # Acción 1: El robot intenta mover la caja (basado en sus creencias)
    robot.act("mover_caja")  # No actuará porque cree que hay un obstáculo
    
    # Evento 2: El robot percibe que no hay obstáculo (actualiza sus creencias)
    robot.update_beliefs("no_ver_obstaculo")
    
    # Acción 2: El robot intenta mover la caja nuevamente
    robot.act("mover_caja")  # Puede fallar si el estado real del mundo es distinto