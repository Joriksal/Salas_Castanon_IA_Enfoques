from collections import defaultdict  # Para crear diccionarios con valores por defecto, útiles para almacenar hechos y relaciones en el conocimiento de fondo.

# Clase que implementa el algoritmo FOIL para aprender reglas lógicas.
class FOIL:
    def __init__(self, target_predicate):
        # Predicado objetivo que se desea aprender (por ejemplo, "enfermo(X)").
        self.target = target_predicate
        # Lista para almacenar las reglas aprendidas.
        self.rules = []

    # Método principal para entrenar el modelo FOIL.
    def fit(self, examples, background_knowledge):
        # Divide los ejemplos en positivos y negativos.
        positives = [ex[0] for ex in examples if ex[1]]  # Ejemplos positivos.
        negatives = [ex[0] for ex in examples if not ex[1]]  # Ejemplos negativos.

        # Itera mientras haya ejemplos positivos que no estén cubiertos por las reglas aprendidas.
        while positives:
            # Aprende una nueva regla que cubra algunos ejemplos positivos.
            rule = self._learn_rule(positives, negatives, background_knowledge)
            if not rule:
                break  # Si no se puede aprender una regla, se detiene.
            self.rules.append(rule)  # Agrega la nueva regla a la lista de reglas.
            # Elimina los ejemplos positivos que ya están cubiertos por la nueva regla.
            positives = [ex for ex in positives if not self._covers(rule, ex, background_knowledge)]

    # Método para aprender una regla lógica.
    def _learn_rule(self, positives, negatives, bk):
        rule = []  # Inicializa la regla como una lista vacía.
        covered_pos = set(positives)  # Ejemplos positivos cubiertos por la regla.
        covered_neg = set(negatives)  # Ejemplos negativos cubiertos por la regla.

        # Itera mientras haya ejemplos negativos cubiertos por la regla parcial.
        while covered_neg:
            best_literal = None  # Literal que maximiza la cobertura de positivos sin cubrir negativos.
            best_score = -1  # Puntaje del mejor literal.

            # Itera sobre los predicados y hechos del conocimiento de fondo.
            for predicate, facts in bk.items():
                for fact in facts:
                    var = 'X'  # Variable genérica para los ejemplos.
                    # Genera un literal basado en los hechos y los ejemplos positivos.
                    if fact[0] in covered_pos:
                        literal = (predicate, fact[1])
                    elif fact[1] in covered_pos:
                        literal = (predicate, fact[0])
                    else:
                        continue

                    # Calcula cuántos positivos y negativos son cubiertos por el literal.
                    pos_match = sum(self._match(example, literal, bk) for example in covered_pos)
                    neg_match = sum(self._match(example, literal, bk) for example in covered_neg)

                    # Selecciona el literal que cubra más positivos sin cubrir negativos.
                    if pos_match > 0 and neg_match == 0:
                        score = pos_match
                        if score > best_score:
                            best_score = score
                            best_literal = literal

            if not best_literal:
                break  # Si no se encuentra un literal válido, se detiene.

            # Agrega el mejor literal a la regla.
            rule.append(best_literal)
            # Actualiza los ejemplos positivos y negativos cubiertos por la regla parcial.
            covered_pos = {ex for ex in covered_pos if self._match(ex, best_literal, bk)}
            covered_neg = {ex for ex in covered_neg if self._match(ex, best_literal, bk)}

        return rule if rule else None  # Retorna la regla aprendida o None si no se pudo aprender.

    # Verifica si un literal se cumple para un ejemplo dado.
    def _match(self, example, literal, bk):
        pred, val = literal  # Descompone el literal en predicado y valor.
        if pred not in bk:
            return False  # Si el predicado no está en el conocimiento de fondo, retorna False.
        # Verifica si el literal se cumple en el conocimiento de fondo.
        return (example, val) in bk[pred] or (val, example) in bk[pred]

    # Verifica si una regla cubre un ejemplo dado.
    def _covers(self, rule, example, bk):
        # La regla cubre el ejemplo si todos sus literales se cumplen.
        return all(self._match(example, lit, bk) for lit in rule)

    # Representación en texto de las reglas aprendidas.
    def __str__(self):
        out = ""
        for i, rule in enumerate(self.rules, 1):
            # Convierte cada regla en una representación legible.
            conditions = " ∧ ".join(f"{pred}(X,{val})" for pred, val in rule)
            out += f"Regla {i}: enfermo(X) ← {conditions}\n"
        return out if out else "No se encontraron reglas."


# --- EJEMPLO DE USO ---
if __name__ == "__main__":
    # Crea una instancia del algoritmo FOIL para aprender el predicado "enfermo(X)".
    foil = FOIL(("enfermo", "X"))

    # Ejemplos positivos y negativos.
    examples = [
        ("juan", True),  # Juan está enfermo (positivo).
        ("pedro", True),  # Pedro está enfermo (positivo).
        ("maria", False),  # María no está enferma (negativo).
        ("lucia", False)  # Lucía no está enferma (negativo).
    ]

    # Conocimiento de fondo: hechos relacionados con los ejemplos.
    background = {
        "sintoma": [("juan", "fiebre"), ("pedro", "fiebre"), ("maria", "tos")],
        "contacto": [("juan", "pedro"), ("pedro", "maria"), ("lucia", "juan")],
        "edad": [("juan", "30"), ("pedro", "25"), ("maria", "40"), ("lucia", "28")]
    }

    # Entrena el modelo FOIL con los ejemplos y el conocimiento de fondo.
    foil.fit(examples, background)

    # Imprime las reglas aprendidas.
    print("Reglas aprendidas:")
    print(foil)
