class ExpertSystem:
    """
    Clase que implementa un sistema experto basado en reglas.
    Este sistema utiliza un enfoque de encadenamiento hacia adelante para inferir conclusiones
    a partir de hechos observados y reglas predefinidas.
    """

    def __init__(self):
        """
        Inicializa el sistema experto con:
        - Una base de reglas: conjunto de reglas que relacionan antecedentes (condiciones) con conclusiones.
        - Una memoria de trabajo: almacena los hechos observados y las conclusiones inferidas.
        """
        # Base de reglas: cada regla está definida como un par (antecedentes, conclusión).
        # Los antecedentes son un conjunto de condiciones necesarias para inferir la conclusión.
        self.rules = [
            ({"batería_descargada", "faros_debil"}, "alternador_fallo"),  # Si la batería está descargada y los faros son débiles, el alternador falla.
            ({"motor_no_arranca", "sin_combustible"}, "tanque_vacio"),    # Si el motor no arranca y no hay combustible, el tanque está vacío.
            ({"temperatura_alta", "refrigerante_bajo"}, "sobrecalentamiento"),  # Si la temperatura es alta y el refrigerante es bajo, hay sobrecalentamiento.
            ({"alternador_fallo", "sobrecalentamiento"}, "reparacion_urgente")  # Si el alternador falla y hay sobrecalentamiento, se requiere reparación urgente.
        ]
        # Memoria de trabajo: almacena los hechos observados y las conclusiones inferidas.
        # Se utiliza un conjunto (set) para evitar duplicados y facilitar las búsquedas.
        self.working_memory = set()

    def add_fact(self, fact: str):
        """
        Añade un hecho observado a la memoria de trabajo.
        :param fact: Hecho observado (cadena de texto que representa una condición o síntoma).
        """
        # Agrega el hecho al conjunto de la memoria de trabajo.
        self.working_memory.add(fact)

    def forward_chaining(self):
        """
        Realiza el encadenamiento hacia adelante para inferir nuevas conclusiones.
        Este proceso evalúa las reglas para determinar si los antecedentes están presentes
        en la memoria de trabajo. Si se cumplen, se añade la conclusión a la memoria.
        """
        # Bandera que indica si se han inferido nuevos hechos en la iteración actual.
        new_facts = True
        while new_facts:  # Repite mientras se sigan generando nuevos hechos.
            new_facts = False  # Reinicia la bandera al inicio de cada iteración.
            for antecedents, conclusion in self.rules:
                # Verifica si todos los antecedentes de la regla están en la memoria de trabajo
                # y si la conclusión aún no ha sido inferida.
                if antecedents.issubset(self.working_memory) and conclusion not in self.working_memory:
                    # Si la regla se cumple, añade la conclusión a la memoria de trabajo.
                    self.working_memory.add(conclusion)
                    # Muestra un mensaje indicando qué regla fue activada.
                    print(f"Regla activada: {antecedents} → {conclusion}")
                    # Cambia la bandera para indicar que se ha inferido un nuevo hecho.
                    new_facts = True

    def get_conclusions(self):
        """
        Retorna todos los hechos almacenados en la memoria de trabajo.
        Esto incluye tanto los hechos observados como las conclusiones inferidas.
        :return: Conjunto de hechos (memoria de trabajo).
        """
        return self.working_memory


# --- Ejemplo de Uso ---
if __name__ == "__main__":
    """
    Bloque principal que demuestra el uso del sistema experto.
    Aquí se crean instancias, se añaden hechos observados y se ejecuta el encadenamiento hacia adelante.
    """
    # Crea una instancia del sistema experto.
    expert = ExpertSystem()
    
    # Añade hechos observados (síntomas o condiciones iniciales).
    # Estos hechos representan las condiciones iniciales conocidas.
    expert.add_fact("batería_descargada")  # La batería está descargada.
    expert.add_fact("faros_debil")        # Los faros están débiles.
    expert.add_fact("temperatura_alta")   # La temperatura del motor es alta.
    expert.add_fact("refrigerante_bajo")  # El nivel de refrigerante es bajo.
    
    # Ejecuta el encadenamiento hacia adelante para inferir conclusiones.
    print("=== Encadenamiento Hacia Adelante ===")
    expert.forward_chaining()
    
    # Muestra las conclusiones finales inferidas.
    print("\nConclusiones finales:")
    for fact in expert.get_conclusions():
        # Imprime cada hecho almacenado en la memoria de trabajo.
        print(f"- {fact}")