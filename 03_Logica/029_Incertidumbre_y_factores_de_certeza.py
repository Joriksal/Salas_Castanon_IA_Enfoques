# Definición de la clase principal que implementa el sistema de factores de certeza
class CertaintyFactorSystem:
    def __init__(self):
        """
        Constructor de la clase CertaintyFactorSystem.
        Inicializa el sistema con:
        - Una base de reglas, donde cada regla está definida como un par:
          ({antecedentes}, (conclusión, CF)).
          Los antecedentes son un conjunto de hechos necesarios para activar la regla.
          La conclusión es el hecho inferido con un factor de certeza (CF).
        - Un diccionario vacío para almacenar los hechos observados y sus factores de certeza.
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
        Método para añadir un hecho observado al sistema junto con su factor de certeza.
        :param fact: Nombre del hecho (ejemplo: 'fiebre').
        :param cf: Factor de certeza asociado al hecho (valor entre 0 y 1).
        """
        # Se almacena el hecho y su factor de certeza en el diccionario `facts`
        self.facts[fact] = cf

    def infer(self):
        """
        Método para realizar la inferencia lógica en el sistema.
        Combina los factores de certeza de los hechos observados con las reglas definidas
        para generar conclusiones con sus respectivos factores de certeza.
        Utiliza la teoría de Shortliffe para combinar factores de certeza.

        :return: Diccionario con las conclusiones inferidas y sus factores de certeza.
        """
        # Diccionario para almacenar las conclusiones inferidas y sus factores de certeza
        conclusions = {}

        # Iterar sobre cada regla en la base de reglas
        for antecedents, (conclusion, rule_cf) in self.rules:
            # Verificar si todos los antecedentes de la regla están presentes en los hechos observados
            if antecedents.issubset(self.facts.keys()):
                # Calcular el CF de la regla aplicada:
                # Se toma el mínimo de los CFs de los antecedentes y se multiplica por el CF de la regla
                min_cf = min(self.facts[ant] for ant in antecedents)
                conclusion_cf = min_cf * rule_cf

                # Si ya existe un CF para esta conclusión, se combina con el nuevo CF
                if conclusion in conclusions:
                    old_cf = conclusions[conclusion]
                    # Fórmula de combinación de CFs: CF1 + CF2 - (CF1 * CF2)
                    conclusions[conclusion] = old_cf + conclusion_cf - (old_cf * conclusion_cf)
                else:
                    # Si no hay CF previo para esta conclusión, se asigna directamente
                    conclusions[conclusion] = conclusion_cf

        # Retornar el diccionario con las conclusiones inferidas y sus factores de certeza
        return conclusions

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Punto de entrada principal del programa
    # Aquí se crea una instancia del sistema y se ejecuta un ejemplo práctico

    # Crear una instancia del sistema de factores de certeza
    system = CertaintyFactorSystem()
    
    # Añadir hechos observados con sus factores de certeza
    # Ejemplo: fiebre con 60% de certeza, tos con 90%, dolor de garganta con 50%
    system.add_fact("fiebre", 0.6)  # Se registra el hecho "fiebre" con un CF de 0.6
    system.add_fact("tos", 0.9)  # Se registra el hecho "tos" con un CF de 0.9
    system.add_fact("dolor_garganta", 0.5)  # Se registra el hecho "dolor_garganta" con un CF de 0.5
    
    # Realizar la inferencia lógica
    # Se procesan las reglas y los hechos observados para generar conclusiones
    results = system.infer()
    
    # Mostrar las conclusiones inferidas con sus factores de certeza
    print("Conclusiones con Factores de Certeza:")
    for disease, cf in results.items():
        # Se imprime cada conclusión con su probabilidad en porcentaje
        print(f"- {disease}: {cf*100:.1f}% de probabilidad")