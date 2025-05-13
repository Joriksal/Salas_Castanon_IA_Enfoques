# Importamos las librerías necesarias
import numpy as np  # Librería para cálculos matemáticos y manejo de arrays
from collections import defaultdict  # Estructura de datos para manejar diccionarios con valores por defecto
import math  # Librería para operaciones matemáticas avanzadas
from sklearn.feature_extraction.text import TfidfVectorizer  # Herramienta para calcular TF-IDF (no se usa en este código, pero podría ser útil en el futuro)

# Definimos la clase ProbabilisticRetrieval para implementar el modelo de recuperación de información probabilística
class ProbabilisticRetrieval:
    def __init__(self, k1=1.5, b=0.75):
        """
        Inicializa los parámetros del modelo BM25 y las estructuras de datos necesarias.
        k1: Parámetro de saturación de frecuencia (controla la sensibilidad a la frecuencia de términos).
        b: Parámetro de normalización por longitud (controla el impacto de la longitud del documento).
        """
        self.k1 = k1  # Parámetro de ajuste para la frecuencia de términos
        self.b = b  # Parámetro de ajuste para la longitud del documento
        self.corpus = []  # Lista para almacenar los documentos del corpus
        self.vocab = set()  # Conjunto para almacenar el vocabulario único del corpus
        self.doc_lengths = []  # Lista para almacenar la longitud de cada documento
        self.avg_doc_length = 0  # Longitud promedio de los documentos
        self.doc_term_freqs = []  # Lista para almacenar las frecuencias de términos por documento
        self.inverted_index = defaultdict(list)  # Índice invertido para mapear términos a documentos

    def build_index(self, corpus):
        """
        Construye el índice invertido a partir del corpus.
        corpus: Lista de documentos (cada documento es una cadena de texto).
        """
        self.corpus = corpus  # Guardamos el corpus
        # Calculamos la longitud de cada documento (número de palabras)
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        # Calculamos la longitud promedio de los documentos
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # Procesamos cada documento para construir el índice invertido
        for doc_id, doc in enumerate(corpus):
            term_freq = defaultdict(int)  # Diccionario para almacenar la frecuencia de términos en el documento
            for term in doc.split():  # Dividimos el documento en palabras (tokens)
                self.vocab.add(term)  # Añadimos el término al vocabulario
                term_freq[term] += 1  # Incrementamos la frecuencia del término
                self.inverted_index[term].append(doc_id)  # Añadimos el documento al índice invertido para este término
            self.doc_term_freqs.append(term_freq)  # Guardamos las frecuencias de términos del documento

    def bm25_score(self, query, doc_id):
        """
        Calcula el puntaje BM25 para un documento dado una consulta.
        query: Cadena de texto que representa la consulta.
        doc_id: ID del documento a evaluar.
        """
        score = 0  # Inicializamos el puntaje en 0
        doc_length = self.doc_lengths[doc_id]  # Longitud del documento actual
        term_freqs = self.doc_term_freqs[doc_id]  # Frecuencias de términos del documento actual
        
        for term in query.split():  # Iteramos sobre cada término de la consulta
            if term not in self.vocab:  # Si el término no está en el vocabulario, lo ignoramos
                continue
            # Frecuencia del término en el documento
            tf = term_freqs.get(term, 0)
            # Número de documentos que contienen el término
            n = len(self.inverted_index[term])
            # Cálculo del peso IDF (Inverse Document Frequency)
            idf = math.log((len(self.corpus) - n + 0.5) / (n + 0.5) + 1.0)
            # Componente TF normalizado
            numerator = tf * (self.k1 + 1)  # Numerador de la fórmula BM25
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))  # Denominador
            score += idf * (numerator / denominator)  # Sumamos el puntaje del término al puntaje total
        return score  # Devolvemos el puntaje final

    def rank_documents(self, query):
        """
        Rankea los documentos según su relevancia para una consulta.
        query: Cadena de texto que representa la consulta.
        """
        scores = []  # Lista para almacenar los puntajes de cada documento
        for doc_id in range(len(self.corpus)):  # Iteramos sobre todos los documentos
            scores.append((doc_id, self.bm25_score(query, doc_id)))  # Calculamos el puntaje BM25 para cada documento
        # Ordenamos los documentos por puntaje en orden descendente
        return sorted(scores, key=lambda x: x[1], reverse=True)

# --- Ejemplo de Uso ---
# Definimos un corpus de documentos
corpus = [
    "el gato come pescado",  # Documento 0
    "el perro come carne",  # Documento 1
    "el gato y el perro juegan",  # Documento 2
    "la gata come arroz"  # Documento 3
]

# Definimos las consultas que queremos evaluar
queries = ["gato come", "perro"]

# Creamos una instancia de la clase ProbabilisticRetrieval
pr = ProbabilisticRetrieval()
# Construimos el índice invertido a partir del corpus
pr.build_index(corpus)

# Recuperamos y rankeamos los documentos para cada consulta
for query in queries:  # Iteramos sobre cada consulta
    ranked_docs = pr.rank_documents(query)  # Obtenemos los documentos rankeados
    print(f"\nConsulta: '{query}'")  # Mostramos la consulta actual
    for doc_id, score in ranked_docs:  # Iteramos sobre los documentos rankeados
        # Mostramos el ID del documento, su puntaje y el contenido del documento
        print(f"Doc {doc_id}: Score={score:.2f} -> {corpus[doc_id]}")