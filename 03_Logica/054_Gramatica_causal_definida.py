from collections import defaultdict  # Para crear diccionarios con valores por defecto, útiles para almacenar reglas causales donde cada clave tiene una lista de efectos asociados.

class CausalGrammar:
    def __init__(self):
        # Inicializa un diccionario para almacenar las reglas causales.
        # Las claves son las causas y los valores son listas de efectos asociados.
        self.rules = defaultdict(list)  # Diccionario de reglas causales

    def add_rule(self, cause, effect):
        """
        Añade una regla causal a la gramática.
        Parámetros:
        - cause: La causa (evento inicial).
        - effect: El efecto (resultado del evento).
        """
        self.rules[cause].append(effect)

    def infer_effects(self, facts):
        """
        Infere los efectos a partir de un conjunto de hechos iniciales.
        Parámetros:
        - facts: Lista de hechos observados.
        Retorna:
        - Un conjunto de efectos inferidos a partir de las reglas causales.
        """
        effects = set()  # Usamos un conjunto para evitar duplicados.
        for fact in facts:
            # Si el hecho observado tiene reglas asociadas, añadimos los efectos.
            if fact in self.rules:
                effects.update(self.rules[fact])
        return effects

    def explain(self, effect):
        """
        Explica las posibles causas de un efecto dado.
        Parámetros:
        - effect: El efecto que queremos explicar.
        Retorna:
        - Una lista de causas que pueden haber producido el efecto.
        """
        return [cause for cause, effects in self.rules.items() if effect in effects]

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Instancia de la gramática causal
    gcd = CausalGrammar()
    
    # Definición de reglas causales
    # Cada regla define una relación causa-efecto.
    gcd.add_rule("lluvia", "cesped_mojado")  # Si llueve, el césped estará mojado.
    gcd.add_rule("regar_cesped", "cesped_mojado")  # Si se riega el césped, estará mojado.
    gcd.add_rule("lluvia", "cancelar_picnic")  # Si llueve, se cancela el picnic.
    gcd.add_rule("cesped_mojado", "resbalar")  # Si el césped está mojado, alguien puede resbalar.

    # Hechos observados
    facts = ["lluvia"]  # Observamos que está lloviendo.
    
    # Inferencia de efectos
    print(f"Si ocurre: {facts}, entonces:")
    effects = gcd.infer_effects(facts)  # Inferimos los efectos de los hechos observados.
    for effect in effects:
        print(f"- {effect}")  # Mostramos los efectos inferidos.

    # Explicación de un efecto
    print("\nPosibles causas de 'cesped_mojado':")
    for cause in gcd.explain("cesped_mojado"):  # Buscamos las posibles causas del efecto.
        print(f"- {cause}")  # Mostramos las causas posibles.