
class SituationCalculus:
    """
    Clase que implementa un modelo básico de cálculo situacional.
    Permite representar y modificar un estado del mundo mediante acciones.
    """
    def __init__(self):
        """
        Inicializa la situación inicial del mundo.
        - La caja comienza en la posición 'A'.
        - El robot comienza en la posición 'B'.
        """
        self.current_situation = {
            'caja_pos': 'A',  # Posición inicial de la caja
            'robot_pos': 'B'  # Posición inicial del robot
        }

    def apply_action(self, action, *args):
        """
        Aplica una acción para cambiar la situación actual del mundo.
        
        Parámetros:
        - action (str): Nombre de la acción a realizar ('mover_robot' o 'mover_caja').
        - *args: Argumentos adicionales necesarios para la acción (por ejemplo, nueva posición).

        Acciones soportadas:
        - 'mover_robot': Mueve el robot a una nueva posición.
        - 'mover_caja': Mueve la caja a una nueva posición, pero solo si el robot está junto a ella.
        """
        # Crear una copia de la situación actual para modificarla
        new_situation = self.current_situation.copy()
        
        if action == 'mover_robot':
            # Cambiar la posición del robot a la nueva posición especificada
            new_situation['robot_pos'] = args[0]  # args[0] = nueva posición
            print(f"Robot se mueve a {args[0]}")
            
        elif action == 'mover_caja':
            # Verificar si el robot está en la misma posición que la caja
            if self.current_situation['robot_pos'] == self.current_situation['caja_pos']:
                # Cambiar la posición de la caja a la nueva posición especificada
                new_situation['caja_pos'] = args[0]  # args[0] = nueva posición
                print(f"Caja movida a {args[0]}")
            else:
                # Error si el robot no está junto a la caja
                print("Error: El robot no está junto a la caja.")
        
        # Actualizar la situación actual con los cambios realizados
        self.current_situation = new_situation

    def get_situation(self):
        """
        Devuelve la situación actual del mundo.
        
        Retorno:
        - dict: Diccionario que representa la situación actual (posiciones de la caja y el robot).
        """
        return self.current_situation

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Crear una instancia del modelo de cálculo situacional
    world = SituationCalculus()
    
    # Mostrar la situación inicial
    print("Situación inicial:", world.get_situation())
    
    # Acción 1: Intentar mover la caja a la posición 'C' (falla porque el robot no está junto a la caja)
    world.apply_action('mover_caja', 'C')
    
    # Acción 2: Mover el robot a la posición de la caja ('A')
    world.apply_action('mover_robot', 'A')
    print("Situación después de mover robot:", world.get_situation())
    
    # Acción 3: Mover la caja a la posición 'C' (éxito porque el robot está junto a la caja)
    world.apply_action('mover_caja', 'C')
    print("Situación final:", world.get_situation())