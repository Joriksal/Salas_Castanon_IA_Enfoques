# Importamos defaultdict de la librería collections.
# defaultdict es un diccionario que permite inicializar valores por defecto para claves inexistentes.
from collections import defaultdict

class GrammarInduction:
    """
    Clase que implementa un algoritmo básico de inducción gramatical para aprender una gramática regular
    a partir de ejemplos positivos de un lenguaje.
    """

    def __init__(self, examples):
        """
        Constructor de la clase. Inicializa los datos de entrenamiento y las estructuras necesarias
        para la inducción gramatical.

        :param examples: Lista de cadenas positivas del lenguaje a aprender.
        """
        self.examples = examples  # Lista de ejemplos positivos del lenguaje.
        self.grammar = defaultdict(list)  # Diccionario que almacenará la gramática inducida.
        self.alphabet = self._extract_alphabet()  # Alfabeto extraído de los ejemplos.

    def _extract_alphabet(self):
        """
        Extrae el conjunto de símbolos únicos (alfabeto) presentes en los ejemplos de entrenamiento.

        :return: Un conjunto con los símbolos únicos encontrados en los ejemplos.
        """
        # Une todas las cadenas de los ejemplos en una sola y extrae los caracteres únicos.
        return set("".join(self.examples))

    def learn_regular_grammar(self):
        """
        Implementa un algoritmo básico de inducción gramatical basado en RPNI (Regular Positive and Negative Inference).
        Este método genera una gramática regular a partir de los ejemplos positivos.

        :return: Gramática regular inducida representada como un diccionario.
        """
        # Paso 1: Construir el conjunto de todos los prefijos posibles de las cadenas de entrenamiento.
        prefixes = set()  # Conjunto para almacenar los prefijos únicos.
        for word in self.examples:  # Itera sobre cada cadena de los ejemplos.
            for i in range(len(word) + 1):  # Genera prefijos desde la cadena vacía hasta la cadena completa.
                prefixes.add(word[:i])  # Agrega cada prefijo al conjunto.

        # Paso 2: Clasificar los prefijos en estados únicos.
        state_classes = {}  # Diccionario para mapear prefijos a estados únicos.
        for i, prefix in enumerate(prefixes):  # Asigna un estado único a cada prefijo.
            state_classes[prefix] = f"q{i}"  # Los estados se nombran como q0, q1, q2, etc.

        # Paso 3: Inferir las transiciones entre estados.
        for prefix in prefixes:  # Itera sobre cada prefijo.
            for symbol in self.alphabet:  # Itera sobre cada símbolo del alfabeto.
                new_prefix = prefix + symbol  # Genera un nuevo prefijo al agregar el símbolo actual.
                if new_prefix in prefixes:  # Verifica si el nuevo prefijo es válido (existe en el conjunto de prefijos).
                    # Agrega una transición desde el estado actual al nuevo estado.
                    self.grammar[state_classes[prefix]].append(
                        f"{symbol} {state_classes[new_prefix]}"
                    )

        # Paso 4: Añadir producciones finales para las cadenas completas.
        for word in self.examples:  # Itera sobre cada cadena de los ejemplos.
            # Marca los estados finales con una producción que genera la cadena vacía (ε).
            self.grammar[state_classes[word]].append("ε")

        # Devuelve la gramática inducida como un diccionario.
        return self.grammar

    def print_grammar(self):
        """
        Muestra la gramática inducida en formato BNF (Backus-Naur Form).
        """
        # Itera sobre cada no terminal (clave del diccionario) y sus producciones (valores).
        for non_terminal, productions in self.grammar.items():
            # Imprime el no terminal seguido de sus producciones separadas por '|'.
            print(f"{non_terminal} -> {' | '.join(productions)}")


# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Lista de ejemplos positivos del lenguaje a^n b^n (ejemplo: "ab", "aabb", "aaabbb").
    training_data = ["ab", "aabb", "aaabbb"]

    # Crea una instancia de la clase GrammarInduction con los datos de entrenamiento.
    inducer = GrammarInduction(training_data)

    # Llama al método para aprender la gramática regular a partir de los ejemplos.
    grammar = inducer.learn_regular_grammar()

    # Imprime un encabezado para indicar que se mostrará la gramática inducida.
    print("Gramática Inducida:")

    # Llama al método para imprimir la gramática inducida en formato BNF.
    inducer.print_grammar()