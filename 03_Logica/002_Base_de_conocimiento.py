class BaseConocimiento:
    def __init__(self):
        # Inicializa la base de conocimiento con:
        # - Un conjunto vacío de hechos (hechos conocidos como verdaderos).
        # - Una lista vacía de reglas (relaciones entre hechos).
        self.hechos = set()
        self.reglas = []

    def agregar_hecho(self, hecho):
        """Añade un hecho a la base de conocimiento.
        
        Args:
            hecho (str): El hecho que se desea agregar.
        """
        self.hechos.add(hecho)

    def agregar_regla(self, premisas, conclusion):
        """Añade una regla del tipo: Si premisas, entonces conclusión.
        
        Args:
            premisas (list): Lista de hechos que deben cumplirse (condiciones).
            conclusion (str): Hecho que se deduce si las premisas son verdaderas.
        """
        # Convierte las premisas en un conjunto para facilitar la comparación.
        self.reglas.append((set(premisas), conclusion))

    def encadenamiento_adelante(self):
        """Ejecuta el encadenamiento hacia adelante para inferir nuevos hechos.
        
        Este método evalúa las reglas repetidamente:
        - Si todas las premisas de una regla están en los hechos conocidos,
          y la conclusión no está en los hechos, se añade la conclusión.
        - El proceso se repite hasta que no se puedan inferir más hechos nuevos.
        """
        cambio = True  # Variable para rastrear si se han añadido nuevos hechos.
        while cambio:
            cambio = False  # Asume que no habrá cambios en esta iteración.
            for premisas, conclusion in self.reglas:
                # Verifica si todas las premisas están en los hechos conocidos
                # y si la conclusión aún no está en los hechos.
                if premisas.issubset(self.hechos) and conclusion not in self.hechos:
                    self.hechos.add(conclusion)  # Añade la conclusión como un nuevo hecho.
                    cambio = True  # Indica que se ha encontrado un nuevo hecho.

    def consultar(self, hecho):
        """Consulta si un hecho es verdadero en la base de conocimiento.
        
        Args:
            hecho (str): El hecho que se desea consultar.
        
        Returns:
            bool: True si el hecho está en la base de conocimiento, False en caso contrario.
        """
        return hecho in self.hechos


# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia de la base de conocimiento.
    bc = BaseConocimiento()
    
    # Agregar hechos iniciales a la base de conocimiento.
    bc.agregar_hecho("p")  # Se añade el hecho "p".
    bc.agregar_hecho("q")  # Se añade el hecho "q".
    
    # Agregar reglas a la base de conocimiento.
    # Ejemplo: Si "p" y "q" son verdaderos, entonces "r" es verdadero.
    bc.agregar_regla(["p", "q"], "r")
    # Ejemplo: Si "r" es verdadero, entonces "s" es verdadero.
    bc.agregar_regla(["r"], "s")
    
    # Ejecutar el encadenamiento hacia adelante para inferir nuevos hechos.
    bc.encadenamiento_adelante()
    
    # Consultar si ciertos hechos están en la base de conocimiento.
    print("¿Está 'r' en la BC?", bc.consultar("r"))  # True, porque "p" y "q" implican "r".
    print("¿Está 's' en la BC?", bc.consultar("s"))  # True, porque "r" implica "s".
    print("¿Está 't' en la BC?", bc.consultar("t"))  # False, porque "t" no se deduce de las reglas.