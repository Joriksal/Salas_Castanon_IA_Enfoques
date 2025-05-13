from collections import defaultdict
import math

class NaiveBayesClassifier:
    def __init__(self):
        """
        Inicializa el clasificador Naive Bayes con estructuras de datos necesarias.
        """
        # Conteo de documentos por clase
        self.class_counts = defaultdict(int)
        # Conteo de palabras en general
        self.feature_counts = defaultdict(int)
        # Conteo de palabras por clase
        self.class_feature_counts = defaultdict(lambda: defaultdict(int))
        # Conjunto de palabras únicas (vocabulario)
        self.vocab = set()
        
        # Hiperparámetros
        self.alpha = 1  # Suavizado de Laplace para evitar probabilidades cero
        self.num_features = 0  # Número total de características (palabras únicas)
    
    def train(self, documents, labels):
        """
        Entrena el clasificador Naive Bayes con los documentos y etiquetas proporcionados.
        
        Args:
            documents: Lista de documentos (cada documento es una lista de palabras).
            labels: Lista de etiquetas correspondientes a los documentos.
        """
        for doc, label in zip(documents, labels):
            # Incrementa el conteo de documentos para la clase actual
            self.class_counts[label] += 1
            for word in doc:
                # Incrementa el conteo de la palabra en general
                self.feature_counts[word] += 1
                # Incrementa el conteo de la palabra para la clase actual
                self.class_feature_counts[label][word] += 1
                # Agrega la palabra al vocabulario
                self.vocab.add(word)
        
        # Calcula el número total de características (palabras únicas)
        self.num_features = len(self.vocab)
        # Lista de clases únicas
        self.classes = list(self.class_counts.keys())
    
    def calculate_log_prob(self, doc, label):
        """
        Calcula la probabilidad logarítmica de que un documento pertenezca a una clase,
        asumiendo independencia condicional entre las características.
        
        Args:
            doc: Documento (lista de palabras).
            label: Clase para la cual se calcula la probabilidad.
        
        Returns:
            log_prob: Probabilidad logarítmica del documento dado la clase.
        """
        # Probabilidad a priori de la clase
        log_prob = math.log(self.class_counts[label] / sum(self.class_counts.values()))
        
        # Calcula la probabilidad condicional para cada palabra en el documento
        for word in doc:
            # Frecuencia de la palabra en la clase + suavizado
            word_count = self.class_feature_counts[label].get(word, 0) + self.alpha
            # Total de palabras en la clase + suavizado * vocabulario
            total_words = sum(self.class_feature_counts[label].values()) + self.alpha * self.num_features
            # Suma el logaritmo de la probabilidad condicional
            log_prob += math.log(word_count / total_words)
        
        return log_prob
    
    def predict(self, doc):
        """
        Predice la clase más probable para un documento.
        
        Args:
            doc: Documento (lista de palabras).
        
        Returns:
            predicted_class: Clase predicha para el documento.
            probs: Diccionario con las probabilidades de cada clase.
        """
        log_probs = {}
        # Calcula la probabilidad logarítmica para cada clase
        for label in self.classes:
            log_probs[label] = self.calculate_log_prob(doc, label)
        
        # Encuentra el máximo logaritmo de probabilidad para evitar underflow
        max_log_prob = max(log_probs.values())
        probs = {}
        # Convierte log-probs a probabilidades normales
        for label in log_probs:
            probs[label] = math.exp(log_probs[label] - max_log_prob)
        
        # Normaliza las probabilidades para que sumen 1
        total = sum(probs.values())
        for label in probs:
            probs[label] = probs[label] / total if total > 0 else 0
        
        # Encuentra la clase con la mayor probabilidad
        predicted_class = max(probs.items(), key=lambda x: x[1])[0]
        return predicted_class, probs

# Ejemplo de uso corregido
if __name__ == "__main__":
    # Datos de entrenamiento (simplificados)
    train_docs = [
        ["fútbol", "partido", "gol", "equipo", "jugador"],  # deportes
        ["balón", "cancha", "árbitro", "tiempo"],           # deportes
        ["presidente", "congreso", "ley", "gobierno"],      # política
        ["elección", "voto", "partido", "político"]         # política
    ]
    train_labels = ["deportes", "deportes", "política", "política"]
    
    # Crear y entrenar el clasificador
    nb = NaiveBayesClassifier()
    nb.train(train_docs, train_labels)
    
    # Documento de prueba
    test_doc = ["partido", "gobierno", "ley"]
    predicted_class, probs = nb.predict(test_doc)
    
    # Resultados de la predicción
    print(f"Documento clasificado como: {predicted_class}")
    print("Probabilidades:")
    for cls, prob in probs.items():
        print(f"  {cls}: {prob:.2%}")
    
    # Mostrar evidencia de independencia condicional
    print("\nProbabilidades condicionales P(palabra|clase):")
    words_to_check = ["partido", "gobierno", "ley"]
    for word in words_to_check:
        for label in nb.classes:
            # Calcula el conteo de la palabra en la clase con suavizado
            count = nb.class_feature_counts[label].get(word, 0) + nb.alpha
            # Calcula el total de palabras en la clase con suavizado
            total = sum(nb.class_feature_counts[label].values()) + nb.alpha * nb.num_features
            # Calcula la probabilidad condicional
            prob = count / total
            print(f"P('{word}'|{label}) = {prob:.4f}")