class SituationCalculus:
    """
    Clase que implementa un modelo básico de cálculo situacional.
    Este modelo permite representar un estado del mundo y modificarlo mediante acciones específicas.
    """

    def __init__(self):
        """
        Constructor de la clase SituationCalculus.
        Inicializa la situación inicial del mundo con las siguientes condiciones:
        - La caja comienza en la posición 'A'.
        - El robot comienza en la posición 'B'.
        """
        # Diccionario que representa el estado inicial del mundo
        self.current_situation = {
            'caja_pos': 'A',  # La posición inicial de la caja es 'A'
            'robot_pos': 'B'  # La posición inicial del robot es 'B'
        }

    def apply_action(self, action, *args):
        """
        Método para aplicar una acción que modifica la situación actual del mundo.

        Parámetros:
        - action (str): Nombre de la acción a realizar. Puede ser:
            - 'mover_robot': Mueve el robot a una nueva posición.
            - 'mover_caja': Mueve la caja a una nueva posición, pero solo si el robot está junto a la caja.
        - *args: Argumentos adicionales necesarios para la acción. En este caso, la nueva posición.

        Comportamiento:
        - Si la acción es 'mover_robot', se actualiza la posición del robot.
        - Si la acción es 'mover_caja', se verifica que el robot esté junto a la caja antes de moverla.
        """
        # Crear una copia de la situación actual para evitar modificarla directamente
        new_situation = self.current_situation.copy()

        if action == 'mover_robot':
            # Acción: Mover el robot
            # args[0] contiene la nueva posición a la que se moverá el robot
            new_situation['robot_pos'] = args[0]
            print(f"Robot se mueve a {args[0]}")  # Mensaje informativo sobre la acción realizada

        elif action == 'mover_caja':
            # Acción: Mover la caja
            # Verificar si el robot está en la misma posición que la caja
            if self.current_situation['robot_pos'] == self.current_situation['caja_pos']:
                # Si el robot está junto a la caja, se permite moverla
                new_situation['caja_pos'] = args[0]  # args[0] contiene la nueva posición de la caja
                print(f"Caja movida a {args[0]}")  # Mensaje informativo sobre la acción realizada
            else:
                # Si el robot no está junto a la caja, no se puede mover
                print("Error: El robot no está junto a la caja.")  # Mensaje de error

        # Actualizar la situación actual con los cambios realizados
        self.current_situation = new_situation

    def get_situation(self):
        """
        Método para obtener la situación actual del mundo.

        Retorno:
        - dict: Un diccionario que representa la situación actual, incluyendo:
            - 'caja_pos': La posición actual de la caja.
            - 'robot_pos': La posición actual del robot.
        """
        return self.current_situation  # Devuelve el estado actual del mundo

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Bloque principal del programa. Este código se ejecuta solo si el archivo se ejecuta directamente.

    # Crear una instancia del modelo de cálculo situacional
    # Esto inicializa el estado del mundo con la caja en 'A' y el robot en 'B'
    world = SituationCalculus()

    # Mostrar la situación inicial del mundo
    # Esto imprimirá el estado inicial: {'caja_pos': 'A', 'robot_pos': 'B'}
    print("Situación inicial:", world.get_situation())

    # Acción 1: Intentar mover la caja a la posición 'C'
    # Esto fallará porque el robot no está junto a la caja
    world.apply_action('mover_caja', 'C')

    # Acción 2: Mover el robot a la posición de la caja ('A')
    # Esto actualizará la posición del robot a 'A'
    world.apply_action('mover_robot', 'A')
    # Mostrar la situación después de mover el robot
    # Esto imprimirá el estado actualizado: {'caja_pos': 'A', 'robot_pos': 'A'}
    print("Situación después de mover robot:", world.get_situation())

    # Acción 3: Mover la caja a la posición 'C'
    # Esto tendrá éxito porque el robot está junto a la caja
    world.apply_action('mover_caja', 'C')
    # Mostrar la situación final del mundo
    # Esto imprimirá el estado final: {'caja_pos': 'C', 'robot_pos': 'A'}
    print("Situación final:", world.get_situation())