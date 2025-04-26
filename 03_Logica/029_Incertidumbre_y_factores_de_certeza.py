class CertaintyFactorSystem:
    def __init__(self):
        """
        Inicializa el sistema con una base de reglas y un diccionario para los hechos observados.
        Las reglas están definidas como pares: ({antecedentes}, (conclusión, CF)).
        """
        # Base de reglas con factores de certeza (CF)
        self.rules = [
            ({"fiebre", "tos"}, ("gripe", 0.8)),  # Si hay fiebre y tos, hay 80% de certeza de gripe
            ({"fiebre", "dolor_garganta"}, ("amigdalitis", 0.7)),  # Fiebre y dolor de garganta → amigdalitis (70%)
            ({"dolor_cabeza", "nauseas"}, ("migraña", 0.9))  # Dolor de cabeza y náuseas → migraña (90%)
        ]
        # Diccionario para almacenar los hechos observados con sus factores de certeza
        self.facts = {}

    def add_fact(self, fact: str, cf: float):
        """
        Añade un hecho observado al sistema junto con su factor de certeza.
        :param fact: Nombre del hecho (ejemplo: 'fiebre').
        :param cf: Factor de certeza asociado al hecho (valor entre 0 y 1).
        """
        self.facts[fact] = cf

    def infer(self):
        """
        Realiza la inferencia lógica combinando los factores de certeza de los hechos observados
        y las reglas definidas en el sistema. Utiliza la teoría de Shortliffe para combinar CFs.
        :return: Diccionario con las conclusiones y sus factores de certeza.
        """
        conclusions = {}  # Almacena las conclusiones inferidas y sus CFs

        # Iterar sobre cada regla en la base de reglas
        for antecedents, (conclusion, rule_cf) in self.rules:
            # Verificar si todos los antecedentes de la regla están presentes en los hechos observados
            if antecedents.issubset(self.facts.keys()):
                # Calcular el CF de la regla aplicada: mínimo de los CFs de los antecedentes * CF de la regla
                min_cf = min(self.facts[ant] for ant in antecedents)
                conclusion_cf = min_cf * rule_cf

                # Combinar CFs si hay múltiples reglas que llevan a la misma conclusión
                if conclusion in conclusions:
                    old_cf = conclusions[conclusion]
                    # Fórmula de combinación: CF1 + CF2 - (CF1 * CF2)
                    conclusions[conclusion] = old_cf + conclusion_cf - (old_cf * conclusion_cf)
                else:
                    # Si no hay CF previo para esta conclusión, se asigna directamente
                    conclusions[conclusion] = conclusion_cf

        return conclusions

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Crear una instancia del sistema de factores de certeza
    system = CertaintyFactorSystem()
    
    # Añadir hechos observados con sus factores de certeza
    # Ejemplo: fiebre con 60% de certeza, tos con 90%, dolor de garganta con 50%
    system.add_fact("fiebre", 0.6)
    system.add_fact("tos", 0.9)
    system.add_fact("dolor_garganta", 0.5)
    
    # Realizar la inferencia lógica
    results = system.infer()
    
    # Mostrar las conclusiones inferidas con sus factores de certeza
    print("Conclusiones con Factores de Certeza:")
    for disease, cf in results.items():
        print(f"- {disease}: {cf*100:.1f}% de probabilidad")