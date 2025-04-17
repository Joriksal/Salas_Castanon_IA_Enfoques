import numpy as np
from collections import defaultdict

class PCFG:
    def __init__(self):
        # Diccionarios para almacenar reglas y sus probabilidades
        self.rules = defaultdict(list)  # Almacena las reglas de producción
        self.probs = defaultdict(list)  # Almacena las probabilidades asociadas a las reglas
    
    def add_rule(self, lhs, rhs, prob):
        """
        Añade una regla de producción con su probabilidad.
        - lhs: lado izquierdo de la regla (no terminal).
        - rhs: lado derecho de la regla (lista de símbolos).
        - prob: probabilidad de la regla.
        """
        self.rules[lhs].append(tuple(rhs))  # Convertir rhs a tupla para consistencia
        self.probs[lhs].append(prob)  # Almacenar la probabilidad
    
    def generate(self, symbol):
        """
        Genera una frase recursivamente a partir de un símbolo dado.
        - symbol: símbolo inicial (puede ser terminal o no terminal).
        """
        if symbol not in self.rules:  # Si el símbolo no tiene reglas, es terminal
            return symbol
        
        # Seleccionar una producción basada en las probabilidades
        rhs_options = self.rules[symbol]  # Opciones de producción para el símbolo
        probs = np.array(self.probs[symbol], dtype=np.float64)  # Probabilidades asociadas
        probs /= probs.sum()  # Normalizar las probabilidades (por seguridad)
        chosen_idx = np.random.choice(len(rhs_options), p=probs)  # Elegir una regla al azar
        chosen_rhs = rhs_options[chosen_idx]  # Obtener la producción seleccionada
        
        # Generar recursivamente cada símbolo en la producción seleccionada
        return ' '.join(self.generate(s) for s in chosen_rhs if s != '')  # Ignorar símbolos vacíos

# --- Uso de la clase PCFG ---
pcfg = PCFG()

# Definición de reglas de producción con sus probabilidades
pcfg.add_rule('S', ['NP', 'VP'], 0.9)  # Una oración (S) puede ser un NP seguido de un VP
pcfg.add_rule('S', ['VP'], 0.1)  # Alternativamente, una oración puede ser solo un VP
pcfg.add_rule('NP', ['Det', 'N'], 0.6)  # Un NP puede ser un determinante seguido de un sustantivo
pcfg.add_rule('NP', ['Pron'], 0.4)  # O un pronombre
pcfg.add_rule('VP', ['V', 'NP'], 0.7)  # Un VP puede ser un verbo seguido de un NP
pcfg.add_rule('VP', ['V'], 0.3)  # O solo un verbo
pcfg.add_rule('Det', ['el'], 1.0)  # Un determinante puede ser "el"
pcfg.add_rule('N', ['gato'], 0.5)  # Un sustantivo puede ser "gato"
pcfg.add_rule('N', ['pescado'], 0.5)  # O "pescado"
pcfg.add_rule('Pron', ['él'], 1.0)  # Un pronombre puede ser "él"
pcfg.add_rule('V', ['come'], 1.0)  # Un verbo puede ser "come"

# Generar 5 frases de ejemplo usando la gramática
print("Ejemplos de frases generadas:")
for _ in range(5):
    print("- ", pcfg.generate('S'))  # Generar una frase a partir del símbolo inicial 'S'