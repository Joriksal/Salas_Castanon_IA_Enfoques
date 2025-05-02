from collections import defaultdict

class GrammarInduction:
    def __init__(self, examples):
        """
        Inicializa la clase con los ejemplos de entrenamiento y prepara las estructuras necesarias.
        
        :param examples: Lista de cadenas positivas del lenguaje a aprender.
        """
        self.examples = examples  # Ejemplos positivos del lenguaje.
        self.grammar = defaultdict(list)  # Gramática inducida representada como un diccionario.
        self.alphabet = self._extract_alphabet()  # Alfabeto extraído de los ejemplos.
        
    def _extract_alphabet(self):
        """
        Extrae los símbolos únicos presentes en los ejemplos de entrenamiento.
        
        :return: Un conjunto con los símbolos únicos.
        """
        return set("".join(self.examples))  # Combina todas las cadenas y extrae los caracteres únicos.
    
    def learn_regular_grammar(self):
        """
        Implementa un algoritmo básico de inducción gramatical basado en RPNI (Regular Positive and Negative Inference).
        
        :return: Gramática regular inducida en formato de diccionario.
        """
        # Paso 1: Construir el conjunto de prefijos de todas las cadenas de entrenamiento.
        prefixes = set()
        for word in self.examples:
            for i in range(len(word) + 1):  # Incluye prefijos vacíos.
                prefixes.add(word[:i])  # Agrega cada prefijo de la cadena al conjunto.
        
        # Paso 2: Clasificar los prefijos en estados únicos.
        state_classes = {}
        for i, prefix in enumerate(prefixes):
            state_classes[prefix] = f"q{i}"  # Asigna un estado único a cada prefijo.
        
        # Paso 3: Inferir las transiciones entre estados.
        for prefix in prefixes:
            for symbol in self.alphabet:  # Itera sobre cada símbolo del alfabeto.
                new_prefix = prefix + symbol  # Genera un nuevo prefijo al agregar el símbolo.
                if new_prefix in prefixes:  # Si el nuevo prefijo es válido (existe en el conjunto de prefijos).
                    # Agrega una transición desde el estado actual al nuevo estado.
                    self.grammar[state_classes[prefix]].append(
                        f"{symbol} {state_classes[new_prefix]}"
                    )
        
        # Paso 4: Añadir producciones finales para las cadenas completas.
        for word in self.examples:
            # Marca los estados finales con una producción que genera la cadena vacía (ε).
            self.grammar[state_classes[word]].append("ε")
        
        return self.grammar  # Devuelve la gramática inducida.
    
    def print_grammar(self):
        """
        Muestra la gramática inducida en formato BNF (Backus-Naur Form).
        """
        for non_terminal, productions in self.grammar.items():
            # Imprime cada no terminal y sus producciones separadas por '|'.
            print(f"{non_terminal} -> {' | '.join(productions)}")

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Ejemplos del lenguaje a^n b^n (ejemplo: "ab", "aabb", "aaabbb").
    training_data = ["ab", "aabb", "aaabbb"]
    
    # Crea una instancia de la clase con los datos de entrenamiento.
    inducer = GrammarInduction(training_data)
    
    # Aprende la gramática regular a partir de los ejemplos.
    grammar = inducer.learn_regular_grammar()
    
    # Imprime la gramática inducida.
    print("Gramática Inducida:")
    inducer.print_grammar()