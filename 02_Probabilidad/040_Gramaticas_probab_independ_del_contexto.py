# --- Importación de librerías necesarias ---
import numpy as np  # Librería para operaciones matemáticas avanzadas, como manejo de arreglos y generación de números aleatorios
from collections import defaultdict  # Estructura de datos para crear diccionarios con valores por defecto

# --- Definición de la clase PCFG ---
class PCFG:
    """
    Clase para representar una Gramática Probabilística Independiente del Contexto (PCFG).
    Permite definir reglas de producción con probabilidades asociadas y generar frases basadas en dichas reglas.
    """
    def __init__(self):
        """
        Constructor de la clase PCFG.
        Inicializa los diccionarios para almacenar las reglas de producción y sus probabilidades.
        """
        self.rules = defaultdict(list)  # Diccionario para almacenar las reglas de producción (lado izquierdo -> lista de producciones)
        self.probs = defaultdict(list)  # Diccionario para almacenar las probabilidades asociadas a cada regla

    def add_rule(self, lhs, rhs, prob):
        """
        Añade una regla de producción a la gramática.
        Parámetros:
        - lhs: Lado izquierdo de la regla (símbolo no terminal).
        - rhs: Lado derecho de la regla (lista de símbolos, terminales o no terminales).
        - prob: Probabilidad de la regla (valor entre 0 y 1).
        """
        self.rules[lhs].append(tuple(rhs))  # Convierte el lado derecho (rhs) en una tupla para garantizar consistencia
        self.probs[lhs].append(prob)  # Almacena la probabilidad asociada a la regla

    def generate(self, symbol):
        """
        Genera una frase recursivamente a partir de un símbolo dado.
        Si el símbolo es terminal, lo devuelve directamente. Si es no terminal, selecciona una regla de producción
        basada en las probabilidades y genera recursivamente los símbolos del lado derecho.
        Parámetros:
        - symbol: Símbolo inicial (puede ser terminal o no terminal).
        Retorna:
        - Una cadena que representa la frase generada.
        """
        if symbol not in self.rules:  # Si el símbolo no tiene reglas asociadas, es un símbolo terminal
            return symbol  # Devuelve el símbolo directamente (es un terminal)

        # Obtener las opciones de producción para el símbolo no terminal
        rhs_options = self.rules[symbol]  # Lista de posibles producciones (lado derecho)
        probs = np.array(self.probs[symbol], dtype=np.float64)  # Convertir las probabilidades a un arreglo NumPy
        probs /= probs.sum()  # Normalizar las probabilidades para asegurarse de que sumen 1 (por seguridad)

        # Seleccionar una producción al azar basada en las probabilidades
        chosen_idx = np.random.choice(len(rhs_options), p=probs)  # Índice de la producción seleccionada
        chosen_rhs = rhs_options[chosen_idx]  # Producción seleccionada (lado derecho)

        # Generar recursivamente cada símbolo en la producción seleccionada
        return ' '.join(self.generate(s) for s in chosen_rhs if s != '')  # Ignorar símbolos vacíos y concatenar resultados

# --- Uso de la clase PCFG ---
pcfg = PCFG()  # Crear una instancia de la clase PCFG

# Definición de reglas de producción con sus probabilidades
# Cada regla define cómo un símbolo no terminal puede expandirse en otros símbolos (terminales o no terminales)
pcfg.add_rule('S', ['NP', 'VP'], 0.9)  # Una oración (S) puede ser un NP seguido de un VP con probabilidad 0.9
pcfg.add_rule('S', ['VP'], 0.1)  # Alternativamente, una oración puede ser solo un VP con probabilidad 0.1
pcfg.add_rule('NP', ['Det', 'N'], 0.6)  # Un NP puede ser un determinante seguido de un sustantivo con probabilidad 0.6
pcfg.add_rule('NP', ['Pron'], 0.4)  # O un pronombre con probabilidad 0.4
pcfg.add_rule('VP', ['V', 'NP'], 0.7)  # Un VP puede ser un verbo seguido de un NP con probabilidad 0.7
pcfg.add_rule('VP', ['V'], 0.3)  # O solo un verbo con probabilidad 0.3
pcfg.add_rule('Det', ['el'], 1.0)  # Un determinante puede ser "el" con probabilidad 1.0
pcfg.add_rule('N', ['gato'], 0.5)  # Un sustantivo puede ser "gato" con probabilidad 0.5
pcfg.add_rule('N', ['pescado'], 0.5)  # O "pescado" con probabilidad 0.5
pcfg.add_rule('Pron', ['él'], 1.0)  # Un pronombre puede ser "él" con probabilidad 1.0
pcfg.add_rule('V', ['come'], 1.0)  # Un verbo puede ser "come" con probabilidad 1.0

# Generar 5 frases de ejemplo usando la gramática
print("Ejemplos de frases generadas:")  # Encabezado para las frases generadas
for _ in range(5):  # Generar 5 frases
    print("- ", pcfg.generate('S'))  # Generar una frase a partir del símbolo inicial 'S' y mostrarla