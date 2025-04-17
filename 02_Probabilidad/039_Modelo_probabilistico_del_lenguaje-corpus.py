import numpy as np
from collections import defaultdict
import random

class LanguageModel:
    def __init__(self, corpus, n=2):
        """
        Inicializa el modelo de lenguaje.
        :param corpus: Lista de oraciones (listas de palabras) para entrenar el modelo.
        :param n: Orden del n-grama (por defecto, bigrama: n=2).
        """
        self.n = n  # Orden del n-grama
        self.counts = defaultdict(lambda: defaultdict(int))  # Almacena las frecuencias de los n-gramas
        self.vocab = set()  # Conjunto de palabras únicas en el corpus
        self.train(corpus)  # Entrena el modelo con el corpus
    
    def train(self, corpus):
        """
        Entrena el modelo de lenguaje calculando las frecuencias de los n-gramas.
        :param corpus: Lista de oraciones (listas de palabras).
        """
        for sentence in corpus:
            # Agrega tokens especiales para inicio (<s>) y fin (</s>) de oración
            tokens = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            for i in range(len(tokens) - self.n + 1):
                # Divide en contexto (n-1 palabras) y palabra objetivo
                context = tuple(tokens[i:i + self.n - 1])
                word = tokens[i + self.n - 1]
                self.counts[context][word] += 1  # Incrementa la frecuencia del n-grama
                self.vocab.add(word)  # Agrega la palabra al vocabulario
    
    def probability(self, context, word, laplace=1):
        """
        Calcula la probabilidad condicional P(word | context) usando suavizado de Laplace.
        :param context: Contexto (n-1 palabras).
        :param word: Palabra objetivo.
        :param laplace: Parámetro de suavizado de Laplace (por defecto, 1).
        :return: Probabilidad condicional.
        """
        context_counts = sum(self.counts[context].values())  # Total de ocurrencias del contexto
        word_count = self.counts[context].get(word, 0)  # Frecuencia de la palabra en el contexto
        # Aplica la fórmula de suavizado de Laplace
        return (word_count + laplace) / (context_counts + laplace * len(self.vocab))
    
    def generate_text(self, max_length=10, seed_context=None):
        """
        Genera texto basado en el modelo entrenado.
        :param max_length: Longitud máxima del texto generado.
        :param seed_context: Contexto inicial para comenzar la generación.
        :return: Texto generado como una cadena.
        """
        if not seed_context:
            # Si no se proporciona contexto inicial, usa tokens de inicio (<s>)
            seed_context = ("<s>",) * (self.n - 1)
        
        context = list(seed_context)  # Convierte el contexto inicial en una lista
        output = list(context)  # Inicializa la salida con el contexto
        
        for _ in range(max_length):
            # Calcula las probabilidades de las palabras siguientes
            next_word_probs = {
                word: self.probability(tuple(context), word)
                for word in self.vocab
            }
            # Selecciona la siguiente palabra basada en las probabilidades
            next_word = random.choices(
                list(next_word_probs.keys()),
                weights=list(next_word_probs.values()),
                k=1
            )[0]
            
            if next_word == "</s>":  # Detiene la generación si se alcanza el token de fin
                break
                
            output.append(next_word)  # Agrega la palabra generada a la salida
            context = output[-(self.n - 1):]  # Actualiza el contexto para el siguiente paso
        
        return " ".join(output)  # Devuelve el texto generado como una cadena

# --- Ejemplo de Uso ---
corpus = [
    ["el", "gato", "come", "pescado"],
    ["la", "gata", "come", "arroz"],
    ["el", "perro", "come", "carne"]
]

# 1. Entrenar modelo (bigramas)
lm = LanguageModel(corpus, n=2)

# 2. Calcular probabilidad P("come" | "el")
context = ("el",)  # Contexto: "el"
word = "come"  # Palabra objetivo: "come"
print(f"P('{word}' | '{' '.join(context)}') = {lm.probability(context, word):.2f}")

# 3. Generar texto
print("\nTexto generado:")
print(lm.generate_text(seed_context=("el",)))  # Genera texto comenzando con "el"