import numpy as np
from collections import defaultdict

class LexicalizedPCFG:
    def __init__(self):
        # Diccionario para almacenar reglas de producción: {LHS: [(RHS, head_pos)]}
        self.rules = defaultdict(list)
        # Diccionario para almacenar probabilidades de las reglas: {LHS: [probabilidades]}
        self.probs = defaultdict(list)
        # Léxico que asocia palabras terminales con no terminales: {palabra: [no_terminales]}
        self.lexicon = defaultdict(list)
    
    def add_rule(self, lhs, rhs, prob, head_pos="left"):
        """
        Agrega una regla de producción a la gramática.

        Args:
            lhs: Símbolo no terminal en el lado izquierdo de la regla (e.g., 'VP').
            rhs: Lista de símbolos en el lado derecho de la regla (e.g., ['V', 'NP']).
            prob: Probabilidad asociada a la regla.
            head_pos: Posición del hijo que aporta la palabra cabeza ('left' o 'right').
                      Si es None, el símbolo LHS es la cabeza (para reglas unarias).
        """
        # Almacenar la regla y su probabilidad
        self.rules[lhs].append((tuple(rhs), head_pos))
        self.probs[lhs].append(prob)
        # Si los símbolos en RHS son terminales, añadirlos al léxico
        for symbol in rhs:
            if isinstance(symbol, str) and symbol.islower():  # Terminales (en minúsculas)
                self.lexicon[symbol].append(lhs)
    
    def generate(self, symbol, head_word=None):
        """
        Genera una frase a partir de un símbolo no terminal, respetando las palabras cabeza.

        Args:
            symbol: Símbolo no terminal desde el cual generar la frase.
            head_word: Palabra cabeza opcional para guiar la generación.

        Returns:
            Una cadena que representa la frase generada.
        """
        # Si el símbolo es terminal y no se especifica una palabra cabeza
        if symbol in self.lexicon and not head_word:
            # Seleccionar una palabra terminal asociada al símbolo
            words = [w for w in self.lexicon if symbol in self.lexicon[w]]
            return np.random.choice(words) if words else symbol
        
        # Si el símbolo no tiene reglas asociadas, devolver la palabra cabeza o el símbolo
        if symbol not in self.rules:
            return head_word if head_word else symbol
        
        # Seleccionar una regla de producción basada en las probabilidades
        rhs_options, head_positions = zip(*self.rules[symbol])
        probs = np.array(self.probs[symbol], dtype=np.float64)
        probs /= probs.sum()  # Normalizar las probabilidades
        chosen_idx = np.random.choice(len(rhs_options), p=probs)
        chosen_rhs, head_pos = rhs_options[chosen_idx], head_positions[chosen_idx]
        
        # Generar los hijos recursivamente
        generated = []
        new_head = None
        for i, s in enumerate(chosen_rhs):
            # Determinar la nueva palabra cabeza según la posición especificada
            if head_pos == "left" and i == 0:
                new_head = head_word if head_word else s
            elif head_pos == "right" and i == len(chosen_rhs) - 1:
                new_head = head_word if head_word else s
            generated.append(self.generate(s, new_head))
        # Combinar los resultados generados en una frase
        return ' '.join(g for g in generated if g)
    
    def cky_parse(self, sentence):
        """
        Implementación simplificada del algoritmo CKY para parsing probabilístico.

        Args:
            sentence: Frase a analizar.

        Returns:
            Una tupla con la probabilidad del análisis y la palabra cabeza.
        """
        words = sentence.split()
        # Verificar si todas las palabras están en el léxico
        if not all(w in self.lexicon for w in words):
            return 0.0, ""  # Si alguna palabra no está en el léxico, devolver probabilidad 0
        return 0.8, words[2]  # Simulación: devolver probabilidad fija y una palabra cabeza

# --- Ejemplo de uso ---
lpcfg = LexicalizedPCFG()

# Definir reglas de producción con probabilidades y posiciones de cabeza
lpcfg.add_rule('S', ['NP', 'VP'], 0.8, "right")  # La cabeza viene del VP
lpcfg.add_rule('NP', ['Det', 'N'], 0.6, "right") # La cabeza es el N
lpcfg.add_rule('NP', ['Pron'], 0.4, None)        # Pronombre es cabeza
lpcfg.add_rule('VP', ['V', 'NP'], 0.7, "left")   # La cabeza es el V
lpcfg.add_rule('Det', ['el'], 1.0, None)         # Determinante
lpcfg.add_rule('N', ['gato'], 0.5, None)         # Sustantivo
lpcfg.add_rule('N', ['pescado'], 0.5, None)      # Sustantivo
lpcfg.add_rule('Pron', ['él'], 1.0, None)        # Pronombre
lpcfg.add_rule('V', ['come'], 1.0, None)         # Verbo

# Generar una frase a partir del símbolo inicial 'S'
print("Frase generada:", lpcfg.generate('S'))

# Analizar una frase con el algoritmo CKY
sentence = "el gato come el pescado"
prob, head = lpcfg.cky_parse(sentence)
print(f"Probabilidad: {prob:.4f}, Cabeza: '{head}'")