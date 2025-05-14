# Clase principal que resuelve ambigüedades en palabras basándose en pistas contextuales
class AmbiguityResolver:
    def __init__(self):
        """
        Constructor de la clase AmbiguityResolver.
        Inicializa una estructura de datos que contiene palabras ambiguas, 
        sus posibles significados, pesos base asociados a cada significado, 
        y pistas contextuales que ajustan dichos pesos.

        Estructura de datos:
        - Cada palabra ambigua tiene una lista de significados posibles.
        - Cada significado tiene:
            - "meaning": El significado textual.
            - "base_weight": Peso base del significado (probabilidad inicial).
            - "clues": Diccionario de pistas contextuales con ajustes de peso.
        """
        self.ambiguous_words = {
            # Palabra ambigua: "banco"
            "banco": [
                # Significado 1: "entidad financiera"
                {
                    "meaning": "entidad_financiera",  # Significado textual
                    "base_weight": 0.6,  # Peso base inicial
                    "clues": {  # Pistas contextuales y ajustes de peso
                        "dinero": 0.3, 
                        "cuenta": 0.2, 
                        "transferencia": 0.2
                    }
                },
                # Significado 2: "mueble"
                {
                    "meaning": "mueble",  # Significado textual
                    "base_weight": 0.4,  # Peso base inicial
                    "clues": {  # Pistas contextuales y ajustes de peso
                        "sentarse": 0.3, 
                        "parque": 0.2, 
                        "madera": 0.1
                    }
                }
            ],
            # Palabra ambigua: "copa"
            "copa": [
                # Significado 1: "trofeo"
                {
                    "meaning": "trofeo",  # Significado textual
                    "base_weight": 0.5,  # Peso base inicial
                    "clues": {  # Pistas contextuales y ajustes de peso
                        "deporte": 0.4, 
                        "ganar": 0.3, 
                        "competencia": 0.2
                    }
                },
                # Significado 2: "vaso"
                {
                    "meaning": "vaso",  # Significado textual
                    "base_weight": 0.5,  # Peso base inicial
                    "clues": {  # Pistas contextuales y ajustes de peso
                        "beber": 0.3, 
                        "vino": 0.2, 
                        "líquido": 0.1
                    }
                }
            ]
        }

    def resolve(self, word, context_clues=[]):
        """
        Resuelve la ambigüedad de una palabra dada un conjunto de pistas contextuales.

        Parámetros:
        - word (str): Palabra ambigua que se desea resolver.
        - context_clues (list): Lista de pistas contextuales relevantes para el significado.

        Retorna:
        - list: Lista de tuplas (significado, peso) ordenadas por relevancia, donde:
            - significado (str): El significado textual.
            - peso (float): Peso calculado basado en las pistas contextuales.

        Proceso:
        1. Obtiene los significados posibles de la palabra desde `self.ambiguous_words`.
        2. Calcula el peso total para cada significado:
            - Comienza con el peso base.
            - Ajusta el peso según las pistas contextuales proporcionadas.
        3. Normaliza el peso para que esté entre 0.0 y 1.0.
        4. Ordena los resultados por peso (descendente) y significado (ascendente).
        """
        # Obtiene los significados posibles de la palabra (en minúsculas para evitar problemas de mayúsculas/minúsculas)
        meanings = self.ambiguous_words.get(word.lower(), [])
        if not meanings:
            # Si la palabra no está registrada, retorna una lista vacía
            return []

        results = []  # Lista para almacenar los resultados (significado, peso)
        for meaning_data in meanings:
            # Inicia con el peso base del significado
            current_weight = meaning_data["base_weight"]
            
            # Ajusta el peso según las pistas contextuales proporcionadas
            for clue in context_clues:
                # Si la pista está en el diccionario de pistas, ajusta el peso
                current_weight += meaning_data["clues"].get(clue, 0)
            
            # Normaliza el peso para que esté en el rango [0.0, 1.0]
            current_weight = max(0.0, min(1.0, current_weight))
            
            # Agrega el significado y su peso calculado a los resultados
            results.append((meaning_data["meaning"], current_weight))
        
        # Ordena los resultados:
        # - Primero por peso en orden descendente (-x[1] invierte el orden).
        # - Luego por significado en orden ascendente (x[0]).
        return sorted(results, key=lambda x: (-x[1], x[0]))

# --- Uso del resolver de ambigüedades ---
if __name__ == "__main__":
    # Crea una instancia de la clase AmbiguityResolver
    resolver = AmbiguityResolver()
    
    # Test 1: Resolviendo la palabra "banco" con contexto financiero
    print("🔍 'banco' con ['dinero', 'cuenta']:")
    for meaning, weight in resolver.resolve("banco", ["dinero", "cuenta"]):
        # Imprime cada significado con su peso en porcentaje
        print(f"- {meaning.ljust(20)} {weight:.0%}")

    # Test 2: Resolviendo la palabra "copa" con un contexto ambiguo
    print("\n🔍 'copa' con ['vino', 'deporte']:")
    for meaning, weight in resolver.resolve("copa", ["vino", "deporte"]):
        # Imprime cada significado con su peso en porcentaje
        print(f"- {meaning.ljust(20)} {weight:.0%}")