# Importamos la librería numpy, que es útil para realizar cálculos numéricos avanzados.
# Aunque no se utiliza explícitamente en este código, podría ser útil para futuras mejoras.
import numpy as np

# Importamos defaultdict de la librería collections.
# defaultdict es una subclase de diccionarios que permite inicializar valores por defecto
# automáticamente cuando se accede a una clave inexistente.
from collections import defaultdict

# Definimos una clase llamada ImprovedSMT (Statistical Machine Translation mejorada).
# Esta clase implementa un modelo básico de traducción automática estadística.
class ImprovedSMT:
    def __init__(self):
        # Inicializamos un diccionario para almacenar traducciones de frases.
        # Utilizamos defaultdict para que cada clave nueva tenga como valor por defecto otro diccionario.
        self.phrase_table = defaultdict(dict)
        
        # Inicializamos un diccionario para almacenar traducciones de palabras individuales.
        self.word_translation = defaultdict(dict)
    
    # Método para entrenar el modelo con oraciones fuente (src_sentences) y objetivo (tgt_sentences).
    def train(self, src_sentences, tgt_sentences):
        """Entrena el modelo utilizando alineamiento básico de palabras y frases."""
        # Llama a un método privado para aprender traducciones de palabras individuales.
        self._learn_word_translations(src_sentences, tgt_sentences)
        
        # Llama a otro método privado para aprender traducciones de frases basadas en las traducciones de palabras.
        self._learn_phrases(src_sentences, tgt_sentences)
    
    # Método privado para aprender traducciones de palabras individuales.
    def _learn_word_translations(self, src_sentences, tgt_sentences):
        """Aprende traducciones de palabras individuales basadas en alineamiento simple."""
        # Iteramos sobre pares de oraciones fuente y objetivo.
        for src, tgt in zip(src_sentences, tgt_sentences):
            # Dividimos las oraciones en palabras utilizando el método split().
            src_words = src.split()  # Palabras de la oración fuente.
            tgt_words = tgt.split()  # Palabras de la oración objetivo.
            
            # Alineamos palabras en el mismo orden (alineamiento simple).
            for s, t in zip(src_words, tgt_words):
                # Si la palabra fuente no tiene traducción registrada, inicializamos el conteo.
                if s not in self.word_translation or t not in self.word_translation[s]:
                    self.word_translation[s][t] = 1
                else:
                    # Si ya existe, incrementamos el conteo.
                    self.word_translation[s][t] += 1
        
        # Normalizamos los conteos para convertirlos en probabilidades.
        for s in self.word_translation:
            total = sum(self.word_translation[s].values())  # Suma total de conteos para la palabra fuente.
            for t in self.word_translation[s]:
                self.word_translation[s][t] /= total  # Dividimos cada conteo por el total.
    
    # Método privado para aprender traducciones de frases.
    def _learn_phrases(self, src_sentences, tgt_sentences):
        """Aprende traducciones de frases basadas en traducciones confiables de palabras."""
        # Iteramos sobre pares de oraciones fuente y objetivo.
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_words = src.split()  # Palabras de la oración fuente.
            tgt_words = tgt.split()  # Palabras de la oración objetivo.
            
            # Generamos pares de frases de 1 a 2 palabras.
            for i in range(len(src_words)):
                for j in range(i+1, min(i+3, len(src_words)+1)):  # Frases de 1-2 palabras.
                    src_phrase = ' '.join(src_words[i:j])  # Frase fuente.
                    tgt_phrase = ' '.join(tgt_words[i:j])  # Frase objetivo.
                    
                    # Verificamos si las palabras individuales tienen alta probabilidad de traducción.
                    valid = True
                    for s, t in zip(src_words[i:j], tgt_words[i:j]):
                        if self.word_translation.get(s, {}).get(t, 0) < 0.5:
                            valid = False
                            break
                    
                    if valid:
                        # Si la frase es válida, la agregamos a la tabla de frases.
                        if tgt_phrase not in self.phrase_table[src_phrase]:
                            self.phrase_table[src_phrase][tgt_phrase] = 1
                        else:
                            self.phrase_table[src_phrase][tgt_phrase] += 1
        
        # Normalizamos los conteos de frases para convertirlos en probabilidades.
        for src in self.phrase_table:
            total = sum(self.phrase_table[src].values())  # Suma total de conteos para la frase fuente.
            for tgt in self.phrase_table[src]:
                self.phrase_table[src][tgt] /= total  # Dividimos cada conteo por el total.

    # Método para traducir una oración utilizando el modelo entrenado.
    def translate(self, sentence):
        """Traduce una oración utilizando el modelo entrenado."""
        words = sentence.split()  # Dividimos la oración en palabras.
        translation = []  # Lista para almacenar la traducción.
        i = 0
        while i < len(words):
            # Intentamos encontrar la frase más larga posible.
            for length in range(min(3, len(words)-i), 0, -1):
                phrase = ' '.join(words[i:i+length])  # Generamos una frase.
                if phrase in self.phrase_table:
                    # Seleccionamos la traducción más probable.
                    best_tgt = max(self.phrase_table[phrase].items(), key=lambda x: x[1])[0]
                    translation.append(best_tgt)
                    i += length  # Avanzamos el índice.
                    break
            else:
                # Si no encontramos una frase, traducimos palabra por palabra.
                word = words[i]
                if word in self.word_translation:
                    best_tgt = max(self.word_translation[word].items(), key=lambda x: x[1])[0]
                    translation.append(best_tgt)
                else:
                    translation.append(word)  # Dejamos la palabra sin traducir.
                i += 1
        return ' '.join(translation)  # Unimos las palabras traducidas en una oración.

# Datos de entrenamiento: pares de oraciones en español (fuente) e inglés (objetivo).
src_sentences = [
    "el gato come pescado",
    "la casa es grande",
    "el perro juega en el parque",
    "la niña lee un libro",
    "el sol brilla fuerte"
]
tgt_sentences = [
    "the cat eats fish",
    "the house is big",
    "the dog plays in the park",
    "the girl reads a book",
    "the sun shines brightly"
]

# Creamos una instancia de la clase ImprovedSMT.
smt = ImprovedSMT()

# Entrenamos el modelo con los datos de entrenamiento.
smt.train(src_sentences, tgt_sentences)

# Imprimimos algunas traducciones aprendidas.
print("Traducciones aprendidas:")
print("el gato ->", smt.phrase_table.get("el gato", {}))  # Traducción de la frase "el gato".
print("perro ->", smt.word_translation.get("perro", {}))  # Traducción de la palabra "perro".

# Frases de prueba para traducir.
test_phrases = [
    "el gato come",
    "la niña lee",
    "el perro juega"
]

# Imprimimos las traducciones de las frases de prueba.
print("\nPruebas de traducción:")
for phrase in test_phrases:
    print(f"{phrase} -> {smt.translate(phrase)}")