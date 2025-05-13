from typing import List, Set, Tuple

class DefaultLogicSystem:
    def __init__(self):
        # Conjunto de hechos conocidos (hard facts)
        self.facts: Set[str] = set()
        # Lista de reglas por defecto: (prerrequisito, justificaciones, conclusión)
        self.default_rules: List[Tuple[str, List[str], str]] = []
        # Conjunto de restricciones: pares de proposiciones incompatibles
        self.constraints: Set[Tuple[str, str]] = set()

    def add_fact(self, fact: str):
        """Agrega un hecho al conjunto de hechos conocidos."""
        self.facts.add(fact)
        self._validate_extension()

    def add_default(self, prerequisite: str, justifications: List[str], conclusion: str):
        """Agrega una regla por defecto al sistema."""
        self.default_rules.append((prerequisite, justifications, conclusion))
        self._validate_extension()

    def add_constraint(self, prop1: str, prop2: str):
        """Define dos proposiciones como incompatibles."""
        self.constraints.add((prop1, prop2))
        self.constraints.add((prop2, prop1))
        self._validate_extension()

    def _validate_extension(self) -> bool:
        """Valida que no haya conflictos entre los hechos actuales y las restricciones."""
        for prop1, prop2 in self.constraints:
            if prop1 in self.facts and prop2 in self.facts:
                return False  # Hay un conflicto
        return True

    def get_extensions(self) -> List[Set[str]]:
        """
        Calcula todas las extensiones posibles usando el algoritmo de Reiter.
        Una extensión es un conjunto consistente de hechos que incluye conclusiones derivadas.
        """
        extensions = [self.facts.copy()]  # Comienza con los hechos conocidos
        
        # Procesa cada regla por defecto
        for pre, justs, conc in self.default_rules:
            new_extensions = []
            
            for ext in extensions:
                # Verifica si el prerrequisito está en la extensión y las justificaciones no bloquean
                if pre in ext and all(just not in ext for just in justs):
                    
                    # Crea una nueva extensión candidata
                    candidate = ext.copy()
                    candidate.add(conc)
                    
                    # Verifica que no haya conflictos con las restricciones
                    valid = True
                    for prop1, prop2 in self.constraints:
                        if prop1 in candidate and prop2 in candidate:
                            valid = False
                            break
                    
                    if valid:
                        new_extensions.append(candidate)
            
            # Agrega las nuevas extensiones
            extensions += new_extensions
        
        # Elimina duplicados y extensiones no máximas
        unique_extensions = []
        for ext in extensions:
            if not any(ext < other for other in extensions):  # Extensiones no contenidas en otras
                unique_extensions.append(ext)
        
        return unique_extensions

    def is_credulous(self, proposition: str) -> bool:
        """Verifica si una proposición está en al menos una extensión."""
        return any(proposition in ext for ext in self.get_extensions())

    def is_skeptical(self, proposition: str) -> bool:
        """Verifica si una proposición está en todas las extensiones."""
        extensions = self.get_extensions()
        return all(proposition in ext for ext in extensions) if extensions else False

# ------------------------------------------
# Ejemplo: Sistema de diagnóstico médico
# ------------------------------------------
if __name__ == "__main__":
    print("=== Default Logic System ===")
    system = DefaultLogicSystem()
    
    # 1. Definir reglas por defecto
    # Si algo es un pájaro y no es un pingüino, entonces vuela
    system.add_default(
        prerequisite="bird(X)",
        justifications=["-penguin(X)"],
        conclusion="flies(X)"
    )
    
    # Si algo es un pájaro y no está herido, entonces vuela
    system.add_default(
        prerequisite="bird(X)",
        justifications=["-injured(X)"],
        conclusion="flies(X)"
    )
    
    # Si algo es un pingüino, entonces no vuela
    system.add_default(
        prerequisite="penguin(X)",
        justifications=[],
        conclusion="-flies(X)"
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