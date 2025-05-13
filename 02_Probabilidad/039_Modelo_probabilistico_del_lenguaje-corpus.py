# Importamos defaultdict de la librería collections.
# defaultdict es una subclase de diccionario que permite inicializar valores por defecto
# automáticamente si una clave no existe.
from collections import defaultdict

# Importamos random para generar números aleatorios.
# En este caso, lo usaremos para seleccionar palabras de manera probabilística.
import random

# Definimos la clase LanguageModel, que implementa un modelo probabilístico de lenguaje basado en n-gramas.
class LanguageModel:
    def __init__(self, corpus, n=2):
        """
        Inicializa el modelo de lenguaje.
        :param corpus: Lista de oraciones (listas de palabras) para entrenar el modelo.
        :param n: Orden del n-grama (por defecto, bigrama: n=2).
        """
        # `self.n` almacena el orden del n-grama (por ejemplo, 2 para bigramas, 3 para trigramas).
        self.n = n
        
        # `self.counts` es un defaultdict de defaultdicts que almacena las frecuencias de los n-gramas.
        # Por ejemplo, self.counts[contexto][palabra] almacena cuántas veces aparece "palabra" después de "contexto".
        self.counts = defaultdict(lambda: defaultdict(int))
        
        # `self.vocab` es un conjunto que almacena todas las palabras únicas del corpus.
        # Los conjuntos no permiten duplicados, lo que facilita la gestión del vocabulario.
        self.vocab = set()
        
        # Llamamos al método `train` para entrenar el modelo con el corpus proporcionado.
        self.train(corpus)
    
    def train(self, corpus):
        """
        Entrena el modelo de lenguaje calculando las frecuencias de los n-gramas.
        :param corpus: Lista de oraciones (listas de palabras).
        """
        for sentence in corpus:
            # Agregamos tokens especiales para marcar el inicio (<s>) y fin (</s>) de cada oración.
            # Esto ayuda al modelo a identificar los límites de las oraciones.
            tokens = ["<s>"] * (self.n - 1) + sentence + ["</s>"]
            
            # Iteramos sobre los tokens para construir los n-gramas.
            for i in range(len(tokens) - self.n + 1):
                # `context` representa el contexto (las primeras n-1 palabras del n-grama).
                context = tuple(tokens[i:i + self.n - 1])
                
                # `word` es la palabra objetivo (la última palabra del n-grama).
                word = tokens[i + self.n - 1]
                
                # Incrementamos la frecuencia del n-grama en `self.counts`.
                self.counts[context][word] += 1
                
                # Agregamos la palabra al vocabulario.
                self.vocab.add(word)
    
    def probability(self, context, word, laplace=1):
        """
        Calcula la probabilidad condicional P(word | context) usando suavizado de Laplace.
        :param context: Contexto (n-1 palabras).
        :param word: Palabra objetivo.
        :param laplace: Parámetro de suavizado de Laplace (por defecto, 1).
        :return: Probabilidad condicional.
        """
        # Calculamos el total de ocurrencias del contexto en el corpus.
        context_counts = sum(self.counts[context].values())
        
        # Obtenemos la frecuencia de la palabra objetivo en el contexto.
        # Si la palabra no existe en el contexto, devolvemos 0.
        word_count = self.counts[context].get(word, 0)
        
        # Aplicamos la fórmula de suavizado de Laplace:
        # (frecuencia de la palabra + laplace) / (total del contexto + laplace * tamaño del vocabulario)
        return (word_count + laplace) / (context_counts + laplace * len(self.vocab))
    
    def generate_text(self, max_length=10, seed_context=None):
        """
        Genera texto basado en el modelo entrenado.
        :param max_length: Longitud máxima del texto generado.
        :param seed_context: Contexto inicial para comenzar la generación.
        :return: Texto generado como una cadena.
        """
        # Si no se proporciona un contexto inicial, usamos tokens de inicio (<s>).
        if not seed_context:
            seed_context = ("<s>",) * (self.n - 1)
        
        # Convertimos el contexto inicial en una lista para facilitar la manipulación.
        context = list(seed_context)
        
        # Inicializamos la salida con el contexto inicial.
        output = list(context)
        
        # Generamos palabras hasta alcanzar la longitud máxima.
        for _ in range(max_length):
            # Calculamos las probabilidades de las palabras siguientes.
            next_word_probs = {
                word: self.probability(tuple(context), word)
                for word in self.vocab
            }
            
            # Seleccionamos la siguiente palabra basada en las probabilidades calculadas.
            # `random.choices` permite seleccionar elementos con pesos específicos.
            next_word = random.choices(
                list(next_word_probs.keys()),  # Lista de palabras posibles.
                weights=list(next_word_probs.values()),  # Pesos (probabilidades).
                k=1  # Seleccionamos una palabra.
            )[0]
            
            # Si la palabra generada es el token de fin (</s>), detenemos la generación.
            if next_word == "</s>":
                break
            
            # Agregamos la palabra generada a la salida.
            output.append(next_word)
            
            # Actualizamos el contexto para el siguiente paso (últimas n-1 palabras).
            context = output[-(self.n - 1):]
        
        # Devolvemos el texto generado como una cadena, uniendo las palabras con espacios.
        return " ".join(output)

# --- Ejemplo de Uso ---

# Definimos un corpus de entrenamiento como una lista de oraciones.
# Cada oración es una lista de palabras.
corpus = [
    ["el", "gato", "come", "pescado"],
    ["la", "gata", "come", "arroz"],
    ["el", "perro", "come", "carne"]
]

# 1. Entrenamos un modelo de lenguaje basado en bigramas (n=2).
lm = LanguageModel(corpus, n=2)

# 2. Calculamos la probabilidad condicional P("come" | "el").
context = ("el",)  # Contexto: "el"
word = "come"  # Palabra objetivo: "come"
print(f"P('{word}' | '{' '.join(context)}') = {lm.probability(context, word):.2f}")

# 3. Generamos texto basado en el modelo entrenado.
print("\nTexto generado:")
print(lm.generate_text(seed_context=("el",)))  # Genera texto comenzando con "el"