import numpy as np
from collections import defaultdict

# --- 1. Clase para Extracción de Entidades (HMM) ---
class HMM_NER:
    def __init__(self, states, observations):
        """
        Inicializa el modelo HMM para la extracción de entidades nombradas.
        :param states: Lista de etiquetas posibles (ej. 'PER', 'LOC', etc.).
        :param observations: Conjunto de palabras únicas en el vocabulario.
        """
        self.states = states  # Etiquetas (ej. 'PER', 'LOC', etc.)
        self.observations = observations  # Palabras únicas en el vocabulario
        self.transition = defaultdict(lambda: defaultdict(float))  # Matriz de transición de estados
        self.emission = defaultdict(lambda: defaultdict(float))   # Matriz de emisión (estado -> palabra)
        self.initial = defaultdict(float)                         # Probabilidades iniciales de los estados
    
    def train(self, labeled_data):
        """
        Entrena el modelo HMM utilizando datos etiquetados.
        :param labeled_data: Lista de pares (palabra, etiqueta).
        """
        prev_state = None
        for (word, state) in labeled_data:
            # Incrementar la cuenta de emisiones para el estado actual
            self.emission[state][word] += 1
            if prev_state is not None:
                # Incrementar la cuenta de transiciones entre estados
                self.transition[prev_state][state] += 1
            else:
                # Incrementar la cuenta de probabilidades iniciales
                self.initial[state] += 1
            prev_state = state
        
        # Normalización de las probabilidades
        for state in self.states:
            # Normalizar las probabilidades de transición
            total_trans = sum(self.transition[state].values())
            if total_trans > 0:
                for next_state in self.transition[state]:
                    self.transition[state][next_state] /= total_trans
            # Normalizar las probabilidades de emisión
            total_emiss = sum(self.emission[state].values())
            if total_emiss > 0:
                for word in self.emission[state]:
                    self.emission[state][word] /= total_emiss
        # Normalizar las probabilidades iniciales
        total_init = sum(self.initial.values())
        if total_init > 0:
            for state in self.initial:
                self.initial[state] /= total_init
    
    def viterbi(self, sequence):
        """
        Algoritmo de Viterbi para encontrar la secuencia más probable de etiquetas.
        :param sequence: Lista de palabras (observaciones).
        :return: Lista de etiquetas correspondientes a la secuencia.
        """
        T = len(sequence)  # Longitud de la secuencia
        N = len(self.states)  # Número de estados
        viterbi = np.zeros((N, T))  # Matriz de probabilidades
        backpointer = np.zeros((N, T), dtype=int)  # Matriz de punteros para backtracking
        
        # Paso de inicialización
        for s in range(N):
            state = self.states[s]
            viterbi[s, 0] = self.initial.get(state, 1e-12) * self.emission[state].get(sequence[0], 1e-12)
        
        # Paso recursivo
        for t in range(1, T):
            for s in range(N):
                state = self.states[s]
                max_prob = -1
                best_prev = 0
                for prev_s in range(N):
                    prev_state = self.states[prev_s]
                    prob = viterbi[prev_s, t-1] * self.transition[prev_state].get(state, 1e-12)
                    if prob > max_prob:
                        max_prob = prob
                        best_prev = prev_s
                viterbi[s, t] = max_prob * self.emission[state].get(sequence[t], 1e-12)
                backpointer[s, t] = best_prev
        
        # Paso de terminación
        best_path = []
        last_state = np.argmax(viterbi[:, -1])  # Estado con mayor probabilidad al final
        best_path.append(last_state)
        
        # Backtracking para reconstruir la secuencia
        for t in range(T-1, 0, -1):
            best_path.insert(0, backpointer[best_path[0], t])
        
        return [self.states[i] for i in best_path]

# --- 2. Clase para Extracción de Relaciones (Naive Bayes) ---
class RelationExtractor:
    def __init__(self, relations):
        """
        Inicializa el modelo Naive Bayes para la extracción de relaciones.
        :param relations: Lista de tipos de relaciones (ej. 'trabaja_en', 'adquiere').
        """
        self.relations = relations  # Tipos de relaciones
        self.word_counts = {rel: defaultdict(int) for rel in relations}  # Conteo de palabras por relación
        self.total_words = {rel: 0 for rel in relations}  # Total de palabras por relación
        self.prior = {rel: 0 for rel in relations}  # Probabilidades a priori de cada relación
    
    def train(self, data):
        """
        Entrena el modelo Naive Bayes con datos etiquetados.
        :param data: Lista de pares (texto, relación).
        """
        for text, rel in data:
            words = text.lower().split()  # Convertir texto a minúsculas y dividir en palabras
            for word in words:
                self.word_counts[rel][word] += 1
                self.total_words[rel] += 1
            self.prior[rel] += 1
        
        # Suavizado de Laplace
        vocab_size = len(set(word for rel in self.word_counts for word in self.word_counts[rel]))
        for rel in self.relations:
            self.prior[rel] /= len(data)  # Normalizar probabilidades a priori
            for word in self.word_counts[rel]:
                self.word_counts[rel][word] = (self.word_counts[rel][word] + 1) / (self.total_words[rel] + vocab_size)
    
    def predict(self, text):
        """
        Predice la relación más probable para un texto dado.
        :param text: Texto de entrada.
        :return: Relación más probable.
        """
        words = text.lower().split()
        scores = {}
        for rel in self.relations:
            scores[rel] = np.log(self.prior[rel])  # Iniciar con el logaritmo de la probabilidad a priori
            for word in words:
                scores[rel] += np.log(self.word_counts[rel].get(word, 1e-12))  # Evitar log(0)
        return max(scores.items(), key=lambda x: x[1])[0]  # Retornar la relación con mayor puntaje

# --- Datos de Entrenamiento Mejorados ---
# Datos etiquetados para extracción de entidades
labeled_sequences = [
    [("Juan", "PER"), ("vive", "O"), ("en", "O"), ("Madrid", "LOC")],
    [("Apple", "ORG"), ("fabrica", "O"), ("iPhones", "PROD")],
    [("El", "O"), ("presidente", "O"), ("de", "O"), ("España", "LOC")],
    [("La", "O"), ("empresa", "O"), ("Microsoft", "ORG"), ("anuncia", "O"), ("Windows", "PROD")]
]

# Datos etiquetados para extracción de relaciones
relation_data = [
    ("Juan trabaja en Google", "trabaja_en"),
    ("Maria trabaja en Amazon", "trabaja_en"),
    ("Microsoft adquiere GitHub", "adquiere"),
    ("Facebook adquiere Instagram", "adquiere"),
    ("Amazon compra Twitch", "adquiere")
]

# --- Entrenamiento y Pruebas ---
# 1. Entrenar HMM para NER
flat_data = [item for seq in labeled_sequences for item in seq]
states = ["PER", "ORG", "LOC", "PROD", "O"]  # Etiquetas posibles
vocab = set(word for seq in labeled_sequences for (word, _) in seq)  # Vocabulario único
hmm = HMM_NER(states, vocab)
hmm.train(flat_data)

# Probar NER
test_sequence = ["El", "CEO", "de", "Microsoft", "viaja", "a", "Nueva", "York"]  # "Nueva York" como dos palabras
predicted_tags = hmm.viterbi(test_sequence)
print("Extracción de Entidades:")
print(list(zip(test_sequence, predicted_tags)))

# 2. Entrenar extractor de relaciones
re = RelationExtractor(relations=["trabaja_en", "adquiere"])
re.train(relation_data)

# Probar relaciones
test_phrases = [
    "Amazon compra Twitch",
    "El CEO trabaja en Apple",
    "Facebook adquiere WhatsApp"
]
print("\nExtracción de Relaciones:")
for phrase in test_phrases:
    print(f"Frase: '{phrase}' -> Relación: {re.predict(phrase)}")