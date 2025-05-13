# Importamos la librería numpy, que se utiliza para realizar cálculos numéricos eficientes, 
# como operaciones con matrices y generación de números aleatorios.
import numpy as np

# Importamos defaultdict de la librería collections. 
# defaultdict es un diccionario que permite inicializar valores por defecto para claves inexistentes.
from collections import defaultdict

# Definimos una clase para representar una Gramática Probabilística Lexicalizada (Lexicalized PCFG).
class LexicalizedPCFG:
    def __init__(self):
        """
        Constructor de la clase. Inicializa las estructuras de datos necesarias para almacenar:
        - Reglas de producción (rules): Diccionario donde las claves son los símbolos no terminales (LHS)
          y los valores son listas de tuplas que contienen el lado derecho de la regla (RHS) y la posición de la cabeza.
        - Probabilidades de las reglas (probs): Diccionario donde las claves son los símbolos no terminales (LHS)
          y los valores son listas de probabilidades asociadas a las reglas.
        - Léxico (lexicon): Diccionario que asocia palabras terminales con los símbolos no terminales que las generan.
        """
        self.rules = defaultdict(list)  # Almacena las reglas de producción
        self.probs = defaultdict(list)  # Almacena las probabilidades de las reglas
        self.lexicon = defaultdict(list)  # Almacena el léxico (terminales y sus no terminales asociados)
    
    def add_rule(self, lhs, rhs, prob, head_pos="left"):
        """
        Agrega una regla de producción a la gramática.

        Args:
            lhs (str): Símbolo no terminal en el lado izquierdo de la regla (e.g., 'VP').
            rhs (list): Lista de símbolos en el lado derecho de la regla (e.g., ['V', 'NP']).
            prob (float): Probabilidad asociada a la regla.
            head_pos (str): Posición del hijo que aporta la palabra cabeza ('left' o 'right').
                            Si es None, el símbolo LHS es la cabeza (para reglas unarias).
        """
        # Guardamos la regla y su probabilidad en los diccionarios correspondientes.
        self.rules[lhs].append((tuple(rhs), head_pos))  # Añadimos la regla al diccionario de reglas
        self.probs[lhs].append(prob)  # Añadimos la probabilidad al diccionario de probabilidades

        # Si los símbolos en RHS son terminales (palabras en minúsculas), los añadimos al léxico.
        for symbol in rhs:
            if isinstance(symbol, str) and symbol.islower():  # Verificamos si es un terminal (minúsculas)
                self.lexicon[symbol].append(lhs)  # Asociamos el terminal con su no terminal en el léxico
    
    def generate(self, symbol, head_word=None):
        """
        Genera una frase a partir de un símbolo no terminal, respetando las palabras cabeza.

        Args:
            symbol (str): Símbolo no terminal desde el cual generar la frase.
            head_word (str): Palabra cabeza opcional para guiar la generación.

        Returns:
            str: Una cadena que representa la frase generada.
        """
        # Caso base: Si el símbolo es terminal y no se especifica una palabra cabeza.
        if symbol in self.lexicon and not head_word:
            # Seleccionamos una palabra terminal asociada al símbolo no terminal.
            words = [w for w in self.lexicon if symbol in self.lexicon[w]]
            return np.random.choice(words) if words else symbol  # Elegimos una palabra aleatoria o devolvemos el símbolo.
        
        # Si el símbolo no tiene reglas asociadas, devolvemos la palabra cabeza o el símbolo.
        if symbol not in self.rules:
            return head_word if head_word else symbol
        
        # Seleccionamos una regla de producción basada en las probabilidades.
        rhs_options, head_positions = zip(*self.rules[symbol])  # Obtenemos las opciones de RHS y posiciones de cabeza.
        probs = np.array(self.probs[symbol], dtype=np.float64)  # Convertimos las probabilidades a un arreglo numpy.
        probs /= probs.sum()  # Normalizamos las probabilidades para que sumen 1.
        chosen_idx = np.random.choice(len(rhs_options), p=probs)  # Elegimos una regla aleatoriamente según las probabilidades.
        chosen_rhs, head_pos = rhs_options[chosen_idx], head_positions[chosen_idx]  # Obtenemos la regla seleccionada.
        
        # Generamos los hijos recursivamente.
        generated = []
        new_head = None
        for i, s in enumerate(chosen_rhs):
            # Determinamos la nueva palabra cabeza según la posición especificada.
            if head_pos == "left" and i == 0:
                new_head = head_word if head_word else s
            elif head_pos == "right" and i == len(chosen_rhs) - 1:
                new_head = head_word if head_word else s
            generated.append(self.generate(s, new_head))  # Generamos recursivamente.
        
        # Combinamos los resultados generados en una frase.
        return ' '.join(g for g in generated if g)
    
    def cky_parse(self, sentence):
        """
        Implementación simplificada del algoritmo CKY para parsing probabilístico.

        Args:
            sentence (str): Frase a analizar.

        Returns:
            tuple: Una tupla con la probabilidad del análisis y la palabra cabeza.
        """
        words = sentence.split()  # Dividimos la frase en palabras.
        # Verificamos si todas las palabras están en el léxico.
        if not all(w in self.lexicon for w in words):
            return 0.0, ""  # Si alguna palabra no está en el léxico, devolvemos probabilidad 0.
        return 0.8, words[2]  # Simulación: devolvemos una probabilidad fija y una palabra cabeza arbitraria.

# --- Ejemplo de uso ---
# Creamos una instancia de la gramática probabilística lexicalizada.
lpcfg = LexicalizedPCFG()

# Definimos reglas de producción con probabilidades y posiciones de cabeza.
lpcfg.add_rule('S', ['NP', 'VP'], 0.8, "right")  # La cabeza viene del VP.
lpcfg.add_rule('NP', ['Det', 'N'], 0.6, "right")  # La cabeza es el N.
lpcfg.add_rule('NP', ['Pron'], 0.4, None)  # Pronombre es cabeza.
lpcfg.add_rule('VP', ['V', 'NP'], 0.7, "left")  # La cabeza es el V.
lpcfg.add_rule('Det', ['el'], 1.0, None)  # Determinante.
lpcfg.add_rule('N', ['gato'], 0.5, None)  # Sustantivo.
lpcfg.add_rule('N', ['pescado'], 0.5, None)  # Sustantivo.
lpcfg.add_rule('Pron', ['él'], 1.0, None)  # Pronombre.
lpcfg.add_rule('V', ['come'], 1.0, None)  # Verbo.

# Generamos una frase a partir del símbolo inicial 'S'.
print("Frase generada:", lpcfg.generate('S'))

# Analizamos una frase con el algoritmo CKY.
sentence = "el gato come el pescado"
prob, head = lpcfg.cky_parse(sentence)
print(f"Probabilidad: {prob:.4f}, Cabeza: '{head}'")