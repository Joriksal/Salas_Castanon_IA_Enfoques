import numpy as np
from collections import defaultdict

class ImprovedSMT:
    def __init__(self):
        # Diccionario para almacenar traducciones de frases
        self.phrase_table = defaultdict(dict)
        # Diccionario para almacenar traducciones de palabras
        self.word_translation = defaultdict(dict)
    
    def train(self, src_sentences, tgt_sentences):
        """Entrena el modelo utilizando alineamiento básico de palabras y frases."""
        # Aprender traducciones de palabras individuales
        self._learn_word_translations(src_sentences, tgt_sentences)
        # Aprender traducciones de frases basadas en las traducciones de palabras
        self._learn_phrases(src_sentences, tgt_sentences)
    
    def _learn_word_translations(self, src_sentences, tgt_sentences):
        """Aprende traducciones de palabras individuales basadas en alineamiento simple."""
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_words = src.split()  # Divide la oración fuente en palabras
            tgt_words = tgt.split()  # Divide la oración objetivo en palabras
            
            # Alineamiento simple: empareja palabras en el mismo orden
            for s, t in zip(src_words, tgt_words):
                if s not in self.word_translation or t not in self.word_translation[s]:
                    # Inicializa el conteo si no existe
                    self.word_translation[s][t] = 1
                else:
                    # Incrementa el conteo si ya existe
                    self.word_translation[s][t] += 1
        
        # Normaliza los conteos a probabilidades
        for s in self.word_translation:
            total = sum(self.word_translation[s].values())
            for t in self.word_translation[s]:
                self.word_translation[s][t] /= total
    
    def _learn_phrases(self, src_sentences, tgt_sentences):
        """Aprende traducciones de frases basadas en traducciones confiables de palabras."""
        for src, tgt in zip(src_sentences, tgt_sentences):
            src_words = src.split()
            tgt_words = tgt.split()
            
            # Genera pares de frases de 1-2 palabras
            for i in range(len(src_words)):
                for j in range(i+1, min(i+3, len(src_words)+1)):  # Frases de 1-2 palabras
                    src_phrase = ' '.join(src_words[i:j])  # Frase fuente
                    tgt_phrase = ' '.join(tgt_words[i:j])  # Frase objetivo
                    
                    # Verifica si las palabras individuales tienen alta probabilidad de traducción
                    valid = True
                    for s, t in zip(src_words[i:j], tgt_words[i:j]):
                        if self.word_translation.get(s, {}).get(t, 0) < 0.5:
                            valid = False
                            break
                    
                    if valid:
                        # Agrega la frase a la tabla de frases
                        if tgt_phrase not in self.phrase_table[src_phrase]:
                            self.phrase_table[src_phrase][tgt_phrase] = 1
                        else:
                            self.phrase_table[src_phrase][tgt_phrase] += 1
        
        # Normaliza los conteos de frases a probabilidades
        for src in self.phrase_table:
            total = sum(self.phrase_table[src].values())
            for tgt in self.phrase_table[src]:
                self.phrase_table[src][tgt] /= total

    def translate(self, sentence):
        """Traduce una oración utilizando el modelo entrenado."""
        words = sentence.split()  # Divide la oración en palabras
        translation = []  # Lista para almacenar la traducción
        i = 0
        while i < len(words):
            # Intenta encontrar la frase más larga posible
            for length in range(min(3, len(words)-i), 0, -1):
                phrase = ' '.join(words[i:i+length])  # Genera una frase
                if phrase in self.phrase_table:
                    # Selecciona la traducción más probable
                    best_tgt = max(self.phrase_table[phrase].items(), key=lambda x: x[1])[0]
                    translation.append(best_tgt)
                    i += length  # Avanza el índice
                    break
            else:
                # Si no encuentra una frase, traduce palabra por palabra
                word = words[i]
                if word in self.word_translation:
                    best_tgt = max(self.word_translation[word].items(), key=lambda x: x[1])[0]
                    translation.append(best_tgt)
                else:
                    translation.append(word)  # Deja la palabra sin traducir
                i += 1
        return ' '.join(translation)  # Une las palabras traducidas en una oración

# Datos de entrenamiento ampliados
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

# Entrenar y probar
smt = ImprovedSMT()
smt.train(src_sentences, tgt_sentences)

# Imprime algunas traducciones aprendidas
print("Traducciones aprendidas:")
print("el gato ->", smt.phrase_table.get("el gato", {}))
print("perro ->", smt.word_translation.get("perro", {}))

# Frases de prueba para traducir
test_phrases = [
    "el gato come",
    "la niña lee",
    "el perro juega"
]

print("\nPruebas de traducción:")
for phrase in test_phrases:
    print(f"{phrase} -> {smt.translate(phrase)}")