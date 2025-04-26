class ExpertSystem:
    def __init__(self):
        """
        Inicializa el sistema experto con una base de reglas y una memoria de trabajo.
        La base de reglas contiene pares de antecedentes (condiciones) y conclusiones.
        La memoria de trabajo almacena los hechos observados.
        """
        # Base de reglas: cada regla tiene un conjunto de antecedentes y una conclusión.
        self.rules = [
            ({"batería_descargada", "faros_debil"}, "alternador_fallo"),  # Si la batería está descargada y los faros son débiles, el alternador falla.
            ({"motor_no_arranca", "sin_combustible"}, "tanque_vacio"),    # Si el motor no arranca y no hay combustible, el tanque está vacío.
            ({"temperatura_alta", "refrigerante_bajo"}, "sobrecalentamiento"),  # Si la temperatura es alta y el refrigerante es bajo, hay sobrecalentamiento.
            ({"alternador_fallo", "sobrecalentamiento"}, "reparacion_urgente")  # Si el alternador falla y hay sobrecalentamiento, se requiere reparación urgente.
        ]
        self.working_memory = set()  # Memoria de trabajo para almacenar los hechos observados.

    def add_fact(self, fact: str):
        """
        Añade un hecho a la memoria de trabajo.
        :param fact: Hecho observado (cadena de texto).
        """
        self.working_memory.add(fact)

    def forward_chaining(self):
        """
        Realiza el encadenamiento hacia adelante para inferir nuevas conclusiones
        basadas en los hechos observados y las reglas definidas.
        """
        new_facts = True  # Bandera para verificar si se han inferido nuevos hechos.
        while new_facts:
            new_facts = False  # Reinicia la bandera.
            for antecedents, conclusion in self.rules:
                # Verifica si los antecedentes de una regla están en la memoria de trabajo
                # y si la conclusión aún no ha sido inferida.
                if antecedents.issubset(self.working_memory) and conclusion not in self.working_memory:
                    # Añade la conclusión a la memoria de trabajo.
                    self.working_memory.add(conclusion)
                    print(f"Regla activada: {antecedents} → {conclusion}")  # Muestra la regla activada.
                    new_facts = True  # Indica que se ha inferido un nuevo hecho.

    def get_conclusions(self):
        """
        Retorna todos los hechos (incluyendo conclusiones) almacenados en la memoria de trabajo.
        :return: Conjunto de hechos.
        """
        return self.working_memory

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Crea una instancia del sistema experto.
    expert = ExpertSystem()
    
    # Añade hechos observados (síntomas o condiciones iniciales).
    expert.add_fact("batería_descargada")
    expert.add_fact("faros_debil")
    expert.add_fact("temperatura_alta")
    expert.add_fact("refrigerante_bajo")
    
    # Ejecuta el encadenamiento hacia adelante para inferir conclusiones.
    print("=== Encadenamiento Hacia Adelante ===")
    expert.forward_chaining()
    
    # Muestra las conclusiones finales inferidas.
    print("\nConclusiones finales:")
    for fact in expert.get_conclusions():
        print(f"- {fact}")