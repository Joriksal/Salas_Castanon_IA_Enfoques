from typing import List, Optional

class Rule:
    """Clase para representar reglas lógicas."""
    def __init__(self, premises: List[str], conclusion: str, weight: float = 1.0):
        self.premises = premises  # Lista de premisas (hechos requeridos para aplicar la regla)
        self.conclusion = conclusion  # Hecho que se infiere si las premisas se cumplen
        self.weight = weight  # Peso de confianza de la regla (opcional)

    def __repr__(self):
        """Representación legible de la regla."""
        return f"Si {' y '.join(self.premises)} → {self.conclusion} (Confianza: {self.weight})"

class KnowledgeBase:
    """Base de conocimiento que almacena hechos y reglas."""
    def __init__(self):
        self.facts: List[str] = []  # Lista de hechos conocidos
        self.rules: List[Rule] = []  # Lista de reglas de inferencia

    def add_rule(self, rule: Rule) -> bool:
        """
        Añade una regla a la base de conocimiento si no está duplicada.
        Retorna True si la regla fue añadida, False si ya existía.
        """
        if not any(r.premises == rule.premises and r.conclusion == rule.conclusion for r in self.rules):
            self.rules.append(rule)
            return True
        return False

    def add_fact(self, fact: str) -> bool:
        """
        Añade un hecho a la base de conocimiento si no existe.
        Retorna True si el hecho fue añadido, False si ya existía.
        """
        if fact not in self.facts:
            self.facts.append(fact)
            return True
        return False

    def forward_chaining(self, verbose: bool = False) -> List[str]:
        """
        Encadenamiento hacia adelante (data-driven).
        Itera sobre las reglas y añade nuevos hechos derivados hasta que no haya más cambios.
        """
        new_facts_added = True  # Bandera para verificar si se añaden nuevos hechos
        iteration = 1  # Contador de iteraciones

        while new_facts_added:
            new_facts_added = False  # Reinicia la bandera en cada iteración
            if verbose:
                print(f"\n*** Iteración {iteration} ***")
                print(f"Hechos actuales: {self.facts}")

            # Recorre todas las reglas
            for rule in self.rules:
                # Verifica si todas las premisas de la regla están en los hechos
                if all(premise in self.facts for premise in rule.premises):
                    # Si la conclusión no está en los hechos, se añade
                    if rule.conclusion not in self.facts:
                        self.add_fact(rule.conclusion)
                        new_facts_added = True  # Marca que se añadió un nuevo hecho
                        if verbose:
                            print(f"Aplicando regla: {rule}")
                            print(f"Nuevo hecho: '{rule.conclusion}'")
            iteration += 1

        return self.facts  # Retorna la lista final de hechos

    def backward_chaining(self, goal: str, visited: Optional[List[str]] = None, verbose: bool = False) -> bool:
        """
        Encadenamiento hacia atrás (goal-driven).
        Intenta demostrar si un objetivo (goal) puede ser derivado a partir de los hechos y reglas.
        """
        if visited is None:
            visited = []  # Lista para evitar ciclos en la búsqueda

        # Si el objetivo ya está en los hechos, se cumple directamente
        if goal in self.facts:
            if verbose:
                print(f"Meta '{goal}' encontrada en hechos.")
            return True

        # Evita ciclos revisando si el objetivo ya fue visitado
        if goal in visited:
            if verbose:
                print(f"Evitando ciclo: '{goal}' ya evaluado.")
            return False

        visited.append(goal)  # Marca el objetivo como visitado

        # Recorre las reglas para encontrar una que concluya el objetivo
        for rule in self.rules:
            if rule.conclusion == goal:
                if verbose:
                    print(f"\nEvaluando regla: {rule}")
                all_premises_met = True  # Bandera para verificar si todas las premisas se cumplen
                for premise in rule.premises:
                    if verbose:
                        print(f"Verificando premisa: '{premise}'")
                    # Llama recursivamente para verificar cada premisa
                    if not self.backward_chaining(premise, visited, verbose):
                        all_premises_met = False
                        break
                if all_premises_met:
                    if verbose:
                        print(f"¡Se cumple la meta '{goal}'!")
                    return True

        return False  # Si no se puede derivar el objetivo, retorna False

# --- Ejemplo: Sistema de Diagnóstico Médico ---
if __name__ == "__main__":
    # Crear la base de conocimiento
    kb = KnowledgeBase()

    # Agregar reglas (sistema experto)
    kb.add_rule(Rule(["fiebre", "dolor de garganta"], "amigdalitis", 0.9))
    kb.add_rule(Rule(["amigdalitis"], "recetar antibióticos", 0.8))
    kb.add_rule(Rule(["fiebre", "tos seca"], "gripe", 0.7))
    kb.add_rule(Rule(["gripe"], "recetar antivirales", 0.6))

    # Hechos iniciales (síntomas del paciente)
    kb.add_fact("fiebre")
    kb.add_fact("dolor de garganta")

    # --- Encadenamiento Hacia Adelante ---
    print("\n=== ENCADENAMIENTO HACIA ADELANTE ===")
    final_facts = kb.forward_chaining(verbose=True)
    print("\nHechos finales derivados:", final_facts)

    # --- Encadenamiento Hacia Atrás ---
    print("\n=== ENCADENAMIENTO HACIA ATRÁS ===")
    goal = "recetar antibióticos"
    result = kb.backward_chaining(goal, verbose=True)
    print(f"\n🔍 ¿Se puede concluir '{goal}'? {'Sí' if result else 'No'}")