import numpy as np
from collections import defaultdict
import math
from sklearn.feature_extraction.text import TfidfVectorizer

class ProbabilisticRetrieval:
    def __init__(self, k1=1.5, b=0.75):
        self.k1 = k1  # Parámetro de saturación de frecuencia
        self.b = b    # Parámetro de normalización por longitud
        self.corpus = []
        self.vocab = set()
        self.doc_lengths = []
        self.avg_doc_length = 0
        self.doc_term_freqs = []
        self.inverted_index = defaultdict(list)
    
    def build_index(self, corpus):
        """Preprocesa el corpus y construye un índice invertido."""
        self.corpus = corpus
        self.doc_lengths = [len(doc.split()) for doc in corpus]
        self.avg_doc_length = np.mean(self.doc_lengths)
        
        # Calcular frecuencia de términos por documento
        for doc_id, doc in enumerate(corpus):
            term_freq = defaultdict(int)
            for term in doc.split():
                self.vocab.add(term)
                term_freq[term] += 1
                self.inverted_index[term].append(doc_id)
            self.doc_term_freqs.append(term_freq)
    
    def bm25_score(self, query, doc_id):
        """Calcula el score BM25 para un documento y una consulta."""
        score = 0
        doc_length = self.doc_lengths[doc_id]
        term_freqs = self.doc_term_freqs[doc_id]
        
        for term in query.split():
            if term not in self.vocab:
                continue
            # Frecuencia del término en el documento
            tf = term_freqs.get(term, 0)
            # Número de documentos que contienen el término
            n = len(self.inverted_index[term])
            # Cálculo del peso IDF
            idf = math.log((len(self.corpus) - n + 0.5) / (n + 0.5) + 1.0)
            # Componente TF normalizado
            numerator = tf * (self.k1 + 1)
            denominator = tf + self.k1 * (1 - self.b + self.b * (doc_length / self.avg_doc_length))
            score += idf * (numerator / denominator)
        return score
    
    def rank_documents(self, query):
        """Rankea documentos según su relevancia a la consulta."""
        scores = []
        for doc_id in range(len(self.corpus)):
            scores.append((doc_id, self.bm25_score(query, doc_id)))
        return sorted(scores, key=lambda x: x[1], reverse=True)

# --- Ejemplo de Uso ---
corpus = [
    "el gato come pescado",
    "el perro come carne",
    "el gato y el perro juegan",
    "la gata come arroz"
]
queries = ["gato come", "perro"]

# Construir índice
pr = ProbabilisticRetrieval()
pr.build_index(corpus)

# Recuperar y rankear documentos para cada consulta
for query in queries:
    ranked_docs = pr.rank_documents(query)
    print(f"\nConsulta: '{query}'")
    for doc_id, score in ranked_docs:
        print(f"Doc {doc_id}: Score={score:.2f} -> {corpus[doc_id]}")