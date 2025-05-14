# Importamos la librería `typing` para usar tipos de datos como List y Optional, 
# lo que ayuda a documentar y verificar los tipos de datos esperados en las funciones.
from typing import List, Optional

# Definición de la clase `Rule`, que representa una regla lógica en el sistema experto.
class Rule:
    """
    Clase para representar reglas lógicas en un sistema experto.
    Cada regla tiene premisas (condiciones necesarias), una conclusión (resultado) 
    y un peso (confianza en la regla).
    """
    def __init__(self, premises: List[str], conclusion: str, weight: float = 1.0):
        """
        Inicializa una regla lógica.
        :param premises: Lista de premisas necesarias para aplicar la regla.
        :param conclusion: Conclusión que se deriva si las premisas se cumplen.
        :param weight: Peso o confianza de la regla (valor entre 0 y 1, opcional).
        """
        self.premises = premises  # Premisas necesarias para aplicar la regla.
        self.conclusion = conclusion  # Conclusión derivada si las premisas se cumplen.
        self.weight = weight  # Peso de confianza de la regla.

    def __repr__(self):
        """
        Representación legible de la regla, útil para depuración y visualización.
        """
        return f"Si {' y '.join(self.premises)} → {self.conclusion} (Confianza: {self.weight})"

# Definición de la clase `KnowledgeBase`, que almacena hechos y reglas.
class KnowledgeBase:
    """
    Base de conocimiento que almacena hechos conocidos y reglas de inferencia.
    Permite realizar encadenamiento hacia adelante y hacia atrás.
    """
    def __init__(self):
        """
        Inicializa la base de conocimiento con listas vacías de hechos y reglas.
        """
        self.facts: List[str] = []  # Lista de hechos conocidos (base de datos de hechos).
        self.rules: List[Rule] = []  # Lista de reglas de inferencia.

    def add_rule(self, rule: Rule) -> bool:
        """
        Añade una regla a la base de conocimiento si no está duplicada.
        :param rule: Regla a añadir.
        :return: True si la regla fue añadida, False si ya existía.
        """
        # Verifica si la regla ya existe comparando premisas y conclusión.
        if not any(r.premises == rule.premises and r.conclusion == rule.conclusion for r in self.rules):
            self.rules.append(rule)  # Añade la regla si no está duplicada.
            return True
        return False  # No añade la regla si ya existe.

    def add_fact(self, fact: str) -> bool:
        """
        Añade un hecho a la base de conocimiento si no existe.
        :param fact: Hecho a añadir.
        :return: True si el hecho fue añadido, False si ya existía.
        """
        if fact not in self.facts:  # Verifica si el hecho ya está en la lista.
            self.facts.append(fact)  # Añade el hecho si no existe.
            return True
        return False  # No añade el hecho si ya existe.

    def forward_chaining(self, verbose: bool = False) -> List[str]:
        """
        Realiza encadenamiento hacia adelante (data-driven).
        Itera sobre las reglas y añade nuevos hechos derivados hasta que no haya más cambios.
        :param verbose: Si es True, imprime información detallada del proceso.
        :return: Lista final de hechos conocidos después del encadenamiento.
        """
        new_facts_added = True  # Bandera para verificar si se añaden nuevos hechos.
        iteration = 1  # Contador de iteraciones para seguimiento.

        while new_facts_added:  # Repite mientras se añadan nuevos hechos.
            new_facts_added = False  # Reinicia la bandera en cada iteración.
            if verbose:
                print(f"\n*** Iteración {iteration} ***")
                print(f"Hechos actuales: {self.facts}")

            # Recorre todas las reglas para verificar si se pueden aplicar.
            for rule in self.rules:
                # Verifica si todas las premisas de la regla están en los hechos conocidos.
                if all(premise in self.facts for premise in rule.premises):
                    # Si la conclusión no está en los hechos, se añade.
                    if rule.conclusion not in self.facts:
                        self.add_fact(rule.conclusion)  # Añade el nuevo hecho.
                        new_facts_added = True  # Marca que se añadió un nuevo hecho.
                        if verbose:
                            print(f"Aplicando regla: {rule}")
                            print(f"Nuevo hecho: '{rule.conclusion}'")
            iteration += 1  # Incrementa el contador de iteraciones.

        return self.facts  # Retorna la lista final de hechos.

    def backward_chaining(self, goal: str, visited: Optional[List[str]] = None, verbose: bool = False) -> bool:
        """
        Realiza encadenamiento hacia atrás (goal-driven).
        Intenta demostrar si un objetivo (goal) puede ser derivado a partir de los hechos y reglas.
        :param goal: Meta u objetivo a demostrar.
        :param visited: Lista de objetivos ya visitados para evitar ciclos.
        :param verbose: Si es True, imprime información detallada del proceso.
        :return: True si el objetivo puede ser derivado, False en caso contrario.
        """
        if visited is None:
            visited = []  # Inicializa la lista de objetivos visitados si no se proporciona.

        # Si el objetivo ya está en los hechos, se cumple directamente.
        if goal in self.facts:
            if verbose:
                print(f"Meta '{goal}' encontrada en hechos.")
            return True

        # Evita ciclos revisando si el objetivo ya fue visitado.
        if goal in visited:
            if verbose:
                print(f"Evitando ciclo: '{goal}' ya evaluado.")
            return False

        visited.append(goal)  # Marca el objetivo como visitado.

        # Recorre las reglas para encontrar una que concluya el objetivo.
        for rule in self.rules:
            if rule.conclusion == goal:  # Si la regla concluye el objetivo.
                if verbose:
                    print(f"\nEvaluando regla: {rule}")
                all_premises_met = True  # Bandera para verificar si todas las premisas se cumplen.
                for premise in rule.premises:
                    if verbose:
                        print(f"Verificando premisa: '{premise}'")
                    # Llama recursivamente para verificar cada premisa.
                    if not self.backward_chaining(premise, visited, verbose):
                        all_premises_met = False  # Si alguna premisa no se cumple, marca como False.
                        break
                if all_premises_met:  # Si todas las premisas se cumplen, el objetivo se cumple.
                    if verbose:
                        print(f"¡Se cumple la meta '{goal}'!")
                    return True

        return False  # Si no se puede derivar el objetivo, retorna False.

# --- Ejemplo: Sistema de Diagnóstico Médico ---
if __name__ == "__main__":
    # Crear la base de conocimiento.
    kb = KnowledgeBase()

    # Agregar reglas al sistema experto.
    kb.add_rule(Rule(["fiebre", "dolor de garganta"], "amigdalitis", 0.9))
    kb.add_rule(Rule(["amigdalitis"], "recetar antibióticos", 0.8))
    kb.add_rule(Rule(["fiebre", "tos seca"], "gripe", 0.7))
    kb.add_rule(Rule(["gripe"], "recetar antivirales", 0.6))

    # Agregar hechos iniciales (síntomas del paciente).
    kb.add_fact("fiebre")
    kb.add_fact("dolor de garganta")

    # --- Encadenamiento Hacia Adelante ---
    print("\n=== ENCADENAMIENTO HACIA ADELANTE ===")
    final_facts = kb.forward_chaining(verbose=True)
    print("\nHechos finales derivados:", final_facts)

    # --- Encadenamiento Hacia Atrás ---
    print("\n=== ENCADENAMIENTO HACIA ATRÁS ===")
    goal = "recetar antibióticos"  # Meta a evaluar.
    result = kb.backward_chaining(goal, verbose=True)
    print(f"\n🔍 ¿Se puede concluir '{goal}'? {'Sí' if result else 'No'}")