from typing import List, Set, Tuple  # Para definir tipos de datos como listas, conjuntos y tuplas, mejorando la legibilidad y el control de tipos.

class DefaultLogicSystem:
    def __init__(self):
        """
        Inicializa el sistema de lógica por defecto con las siguientes estructuras:
        - `facts`: Conjunto de hechos conocidos (hard facts), es decir, proposiciones que se consideran verdaderas.
        - `default_rules`: Lista de reglas por defecto en forma de tuplas (prerrequisito, justificaciones, conclusión).
        - `constraints`: Conjunto de restricciones que definen pares de proposiciones incompatibles.
        """
        self.facts: Set[str] = set()  # Conjunto de hechos conocidos
        self.default_rules: List[Tuple[str, List[str], str]] = []  # Lista de reglas por defecto
        self.constraints: Set[Tuple[str, str]] = set()  # Restricciones entre proposiciones incompatibles

    def add_fact(self, fact: str):
        """
        Agrega un hecho al conjunto de hechos conocidos.
        - Los hechos son proposiciones que se consideran verdaderas sin necesidad de justificación.
        """
        self.facts.add(fact)  # Añadimos el hecho al conjunto
        self._validate_extension()  # Validamos que no haya conflictos con las restricciones

    def add_default(self, prerequisite: str, justifications: List[str], conclusion: str):
        """
        Agrega una regla por defecto al sistema.
        - Una regla por defecto tiene:
          - `prerequisite`: Una condición que debe cumplirse para aplicar la regla.
          - `justifications`: Una lista de proposiciones que no deben estar presentes para aplicar la regla.
          - `conclusion`: La proposición que se deriva si se cumplen las condiciones.
        """
        self.default_rules.append((prerequisite, justifications, conclusion))  # Añadimos la regla
        self._validate_extension()  # Validamos que no haya conflictos con las restricciones

    def add_constraint(self, prop1: str, prop2: str):
        """
        Define dos proposiciones como incompatibles.
        - Si ambas proposiciones están presentes en el mismo conjunto, se considera un conflicto.
        """
        self.constraints.add((prop1, prop2))  # Añadimos la restricción en ambas direcciones
        self.constraints.add((prop2, prop1))
        self._validate_extension()  # Validamos que no haya conflictos con las restricciones

    def _validate_extension(self) -> bool:
        """
        Valida que no haya conflictos entre los hechos actuales y las restricciones.
        - Si dos proposiciones incompatibles están presentes en los hechos, se considera un conflicto.
        """
        for prop1, prop2 in self.constraints:
            if prop1 in self.facts and prop2 in self.facts:
                return False  # Hay un conflicto
        return True  # No hay conflictos

    def get_extensions(self) -> List[Set[str]]:
        """
        Calcula todas las extensiones posibles usando el algoritmo de Reiter.
        - Una extensión es un conjunto consistente de hechos que incluye conclusiones derivadas.
        """
        extensions = [self.facts.copy()]  # Comenzamos con los hechos conocidos como base
        
        # Procesamos cada regla por defecto
        for pre, justs, conc in self.default_rules:
            new_extensions = []  # Lista para almacenar nuevas extensiones
            
            for ext in extensions:
                # Verificamos si el prerrequisito está en la extensión y las justificaciones no bloquean
                if pre in ext and all(just not in ext for just in justs):
                    
                    # Creamos una nueva extensión candidata
                    candidate = ext.copy()
                    candidate.add(conc)  # Añadimos la conclusión derivada
                    
                    # Verificamos que no haya conflictos con las restricciones
                    valid = True
                    for prop1, prop2 in self.constraints:
                        if prop1 in candidate and prop2 in candidate:
                            valid = False
                            break
                    
                    if valid:
                        new_extensions.append(candidate)  # Añadimos la extensión válida
            
            # Agregamos las nuevas extensiones a la lista de extensiones
            extensions += new_extensions
        
        # Eliminamos duplicados y extensiones no máximas
        unique_extensions = []
        for ext in extensions:
            if not any(ext < other for other in extensions):  # Extensiones no contenidas en otras
                unique_extensions.append(ext)
        
        return unique_extensions

    def is_credulous(self, proposition: str) -> bool:
        """
        Verifica si una proposición está en al menos una extensión.
        - Esto representa un razonamiento crédulo (aceptar si es posible).
        """
        return any(proposition in ext for ext in self.get_extensions())

    def is_skeptical(self, proposition: str) -> bool:
        """
        Verifica si una proposición está en todas las extensiones.
        - Esto representa un razonamiento escéptico (aceptar solo si es cierto en todos los casos).
        """
        extensions = self.get_extensions()
        return all(proposition in ext for ext in extensions) if extensions else False

# ------------------------------------------
# Ejemplo: Sistema de lógica por defecto
# ------------------------------------------
if __name__ == "__main__":
    # Mensaje inicial
    print("=== Default Logic System ===")
    system = DefaultLogicSystem()  # Creamos una instancia del sistema
    
    # 1. Definir reglas por defecto
    # Si algo es un pájaro y no es un pingüino, entonces vuela
    system.add_default(
        prerequisite="bird(X)",  # Prerrequisito: ser un pájaro
        justifications=["-penguin(X)"],  # Justificación: no ser un pingüino
        conclusion="flies(X)"  # Conclusión: vuela
    )
    
    # Si algo es un pájaro y no está herido, entonces vuela
    system.add_default(
        prerequisite="bird(X)",  # Prerrequisito: ser un pájaro
        justifications=["-injured(X)"],  # Justificación: no estar herido
        conclusion="flies(X)"  # Conclusión: vuela
    )
    
    # Si algo es un pingüino, entonces no vuela
    system.add_default(
        prerequisite="penguin(X)",  # Prerrequisito: ser un pingüino
        justifications=[],  # Sin justificaciones adicionales
        conclusion="-flies(X)"  # Conclusión: no vuela
    )
    
    # 2. Agregar restricciones
    # Un objeto no puede volar y no volar al mismo tiempo
    system.add_constraint("flies(X)", "-flies(X)")
    
    # 3. Agregar hechos y analizar casos
    print("\nCase 1: Regular bird")
    system.add_fact("bird(tweety)")  # Tweety es un pájaro
    extensions = system.get_extensions()
    print(f"Extensions: {extensions}")
    print(f"Does Tweety fly (credulous)? {system.is_credulous('flies(tweety)')}")
    print(f"Does Tweety fly (skeptical)? {system.is_skeptical('flies(tweety)')}")
    
    print("\nCase 2: Penguin")
    system.add_fact("penguin(opus)")  # Opus es un pingüino
    system.add_fact("bird(opus)")     # Opus también es un pájaro
    extensions = system.get_extensions()
    print(f"Extensions: {extensions}")
    print(f"Does Opus fly (credulous)? {system.is_credulous('flies(opus)')}")
    print(f"Does Opus fly (skeptical)? {system.is_skeptical('flies(opus)')}")
    
    print("\nCase 3: Injured bird")
    system.add_fact("bird(woody)")    # Woody es un pájaro
    system.add_fact("injured(woody)")  # Woody está herido
    extensions = system.get_extensions()
    print(f"Extensions: {extensions}")
    print(f"Does Woody fly (credulous)? {system.is_credulous('flies(woody)')}")
    print(f"Does Woody fly (skeptical)? {system.is_skeptical('flies(woody)')}")