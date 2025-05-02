class AmbiguityResolver:
    def __init__(self):
        """Inicializa el resolver de ambigüedades con una estructura de datos
        que contiene palabras ambiguas, sus significados posibles, pesos base
        y pistas contextuales que ajustan los pesos."""
        self.ambiguous_words = {
            "banco": [
                # Significado 1: "entidad financiera"
                {"meaning": "entidad_financiera", "base_weight": 0.6, 
                 "clues": {"dinero": 0.3, "cuenta": 0.2, "transferencia": 0.2}},
                # Significado 2: "mueble"
                {"meaning": "mueble", "base_weight": 0.4,
                 "clues": {"sentarse": 0.3, "parque": 0.2, "madera": 0.1}}
            ],
            "copa": [
                # Significado 1: "trofeo"
                {"meaning": "trofeo", "base_weight": 0.5, 
                 "clues": {"deporte": 0.4, "ganar": 0.3, "competencia": 0.2}},
                # Significado 2: "vaso"
                {"meaning": "vaso", "base_weight": 0.5,
                 "clues": {"beber": 0.3, "vino": 0.2, "líquido": 0.1}}
            ]
        }

    def resolve(self, word, context_clues=[]):
        """Resuelve la ambigüedad de una palabra dada un conjunto de pistas contextuales.
        
        Parámetros:
        - word: Palabra ambigua a resolver.
        - context_clues: Lista de pistas contextuales relevantes.

        Retorna:
        - Lista de tuplas (significado, peso) ordenadas por relevancia.
        """
        # Obtiene los significados posibles de la palabra
        meanings = self.ambiguous_words.get(word.lower(), [])
        if not meanings:
            return []  # Si no hay significados registrados, retorna vacío

        results = []
        for meaning_data in meanings:
            # Inicia con el peso base del significado
            current_weight = meaning_data["base_weight"]
            
            # Ajusta el peso según las pistas contextuales proporcionadas
            for clue in context_clues:
                current_weight += meaning_data["clues"].get(clue, 0)
            
            # Normaliza el peso para que esté entre 0.0 y 1.0
            current_weight = max(0.0, min(1.0, current_weight))
            results.append((meaning_data["meaning"], current_weight))
        
        # Ordena los resultados por peso (descendente) y significado (ascendente)
        return sorted(results, key=lambda x: (-x[1], x[0]))

# --- Uso del resolver de ambigüedades ---
if __name__ == "__main__":
    resolver = AmbiguityResolver()
    
    # Test 1: Resolviendo "banco" con contexto financiero
    print("🔍 'banco' con ['dinero', 'cuenta']:")
    for meaning, weight in resolver.resolve("banco", ["dinero", "cuenta"]):
        # Imprime cada significado con su peso en porcentaje
        print(f"- {meaning.ljust(20)} {weight:.0%}")

    # Test 2: Resolviendo "copa" con contexto ambiguo
    print("\n🔍 'copa' con ['vino', 'deporte']:")
    for meaning, weight in resolver.resolve("copa", ["vino", "deporte"]):
        # Imprime cada significado con su peso en porcentaje
        print(f"- {meaning.ljust(20)} {weight:.0%}")