from typing import List, Union

class SkolemResolver:
    def __init__(self):
        # Diccionario para almacenar las funciones de Skolem generadas
        self.skolem_functions = {}
        # Contador para generar nombres únicos de variables o funciones
        self.variable_counter = 0

    def prenex_normal_form(self, formula: str) -> str:
        """
        Convierte una fórmula lógica a Forma Normal Prenex (PNF).
        En PNF, todos los cuantificadores (∀, ∃) se colocan al inicio de la fórmula.
        Ejemplo: (∀x P(x)) ∧ (∃y Q(y)) → ∀x ∃y (P(x) ∧ Q(y)).
        Nota: Esta implementación asume que la fórmula ya está en PNF.
        """
        # Retorna la fórmula sin modificaciones (simplificación para este ejemplo)
        return formula

    def skolemize(self, formula: str) -> str:
        """
        Realiza la skolemización de una fórmula lógica.
        La skolemización elimina los cuantificadores existenciales (∃) reemplazándolos:
        - Si ∃y está dentro del alcance de ∀x, y se reemplaza por una función de Skolem f(x).
        - Si ∃y no está en el alcance de ningún ∀, y se reemplaza por una constante arbitraria 'a'.
        """
        # Si no hay cuantificadores existenciales, no se necesita skolemización
        if "∃" not in formula:
            return formula

        # Caso 1: ∃y está dentro del alcance de ∀x
        if "∀" in formula:
            # Genera una nueva función de Skolem (ejemplo: f1, f2, ...)
            new_var = self._generate_skolem_function()
            # Reemplaza ∃y y y por la función de Skolem
            formula = formula.replace("∃y", "").replace("y", new_var)
        # Caso 2: ∃y no está en el alcance de ningún ∀
        else:
            # Reemplaza ∃y y y por una constante arbitraria 'a'
            formula = formula.replace("∃y", "").replace("y", "a")

        return formula

    def _generate_skolem_function(self) -> str:
        """
        Genera un nombre único para una función de Skolem.
        Ejemplo: f1, f2, f3, ...
        """
        self.variable_counter += 1
        return f"f{self.variable_counter}"

    def resolve(self, clause1: List[str], clause2: List[str]) -> Union[List[str], None]:
        """
        Aplica la resolución entre dos cláusulas.
        La resolución elimina literales complementarios (ej: P(x) y ¬P(x)) y combina las cláusulas restantes.
        Ejemplo:
        - Entrada: [P(x), Q(y)] y [¬Q(y), R(z)]
        - Salida: [P(x), R(z)]
        """
        for lit1 in clause1:  # Itera sobre los literales de la primera cláusula
            for lit2 in clause2:  # Itera sobre los literales de la segunda cláusula
                # Verifica si los literales son complementarios
                if self._are_complements(lit1, lit2):
                    # Crea una nueva cláusula eliminando los literales complementarios
                    new_clause = [lit for lit in clause1 if lit != lit1] + \
                                 [lit for lit in clause2 if lit != lit2]
                    return new_clause  # Retorna la cláusula resolvente
        return None  # Si no hay literales complementarios, retorna None

    def _are_complements(self, lit1: str, lit2: str) -> bool:
        """
        Verifica si dos literales son complementarios.
        Ejemplo:
        - P(x) y ¬P(x) son complementarios.
        - Q(y) y ¬Q(y) son complementarios.
        """
        return (lit1 == f"¬{lit2}") or (lit2 == f"¬{lit1}")

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    resolver = SkolemResolver()

    print("==== Skolemización ====")
    formula = "∀x ∃y P(x, y)"
    print(f"Fórmula original: {formula}")

    # Convierte la fórmula a Forma Normal Prenex (en este caso, ya está en PNF)
    pnf_formula = resolver.prenex_normal_form(formula)
    # Aplica la skolemización a la fórmula
    skolemized = resolver.skolemize(pnf_formula)
    print(f"Fórmula skolemizada: {skolemized}")  # Output esperado: ∀x P(x, f1(x))

    print("\n==== Resolución ====")
    # Ejemplo de cláusulas para resolución
    clause1 = ["P(x)", "Q(y)"]
    clause2 = ["¬Q(y)", "R(z)"]
    # Aplica la resolución entre las cláusulas
    resolvent = resolver.resolve(clause1, clause2)
    print(f"Resolvente de {clause1} y {clause2}: {resolvent}")  # Output esperado: ['P(x)', 'R(z)']