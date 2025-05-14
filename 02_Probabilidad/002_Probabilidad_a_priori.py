from collections import defaultdict  # Proporciona un diccionario con valores por defecto.
                                     # Es útil para:
                                     # - Contar las ocurrencias de palabras en el corpus sin necesidad de inicializar manualmente las claves.
                                     # - Evitar errores al acceder a claves inexistentes, ya que devuelve un valor predeterminado (en este caso, 0).

class PriorProbabilityEstimator:
    def __init__(self, laplace_smoothing=1):
        """
        Inicializa el estimador con suavizado de Laplace para manejar casos no vistos.
        
        Args:
            laplace_smoothing (int): Parámetro de suavizado (alpha). 
                                     Por defecto es 1, lo que asegura que ninguna palabra tenga probabilidad 0.
        """
        self.word_counts = defaultdict(int)  # Diccionario para contar las ocurrencias de cada palabra
        self.total_words = 0  # Total de palabras en el corpus
        self.vocabulary = set()  # Conjunto de palabras únicas (vocabulario)
        self.laplace = laplace_smoothing  # Parámetro de suavizado de Laplace
    
    def train(self, corpus):
        """
        Entrena el modelo calculando frecuencias de palabras en el corpus.
        
        Args:
            corpus (list): Lista de oraciones (cada oración es una lista de palabras).
        """
        for sentence in corpus:  # Iterar sobre cada oración en el corpus
            for word in sentence:  # Iterar sobre cada palabra en la oración
                self.word_counts[word] += 1  # Incrementar el contador de la palabra
                self.vocabulary.add(word)  # Agregar la palabra al vocabulario
        self.total_words = sum(self.word_counts.values())  # Calcular el total de palabras en el corpus
    
    def get_prior_probability(self, word):
        """
        Calcula la probabilidad a priori de una palabra con suavizado de Laplace.
        
        Args:
            word (str): Palabra para calcular su probabilidad.
            
        Returns:
            float: Probabilidad estimada P(word).
        """
        count = self.word_counts.get(word, 0)  # Obtener el conteo de la palabra (0 si no existe en el corpus)
        # Aplicar la fórmula de suavizado de Laplace
        return (count + self.laplace) / (self.total_words + self.laplace * len(self.vocabulary))
    
    def get_most_probable_words(self, n=5):
        """
        Devuelve las n palabras más probables según la distribución a priori.
        
        Args:
            n (int): Número de palabras a devolver.
            
        Returns:
            list: Lista de tuplas (palabra, probabilidad).
        """
        # Ordenar las palabras por su frecuencia en orden descendente
        sorted_words = sorted(self.word_counts.items(), 
                            key=lambda x: x[1], 
                            reverse=True)
        # Calcular la probabilidad de cada palabra y devolver las n más probables
        return [(word, count/self.total_words) for word, count in sorted_words[:n]]

# Ejemplo de uso
if __name__ == "__main__":
    # Corpus de ejemplo (normalmente sería mucho más grande)
    training_corpus = [
        ["el", "gato", "come", "pescado"],  # Oración 1
        ["el", "perro", "persigue", "al", "gato"],  # Oración 2
        ["el", "gato", "duerme", "en", "el", "sofá"],  # Oración 3
        ["el", "perro", "come", "carne"]  # Oración 4
    ]
    
    # Crear y entrenar el estimador
    estimator = PriorProbabilityEstimator(laplace_smoothing=1)  # Inicializar con suavizado de Laplace
    estimator.train(training_corpus)  # Entrenar el modelo con el corpus
    
    # Calcular algunas probabilidades a priori
    words_to_check = ["gato", "perro", "elefante", "el"]  # Palabras para calcular su probabilidad
    print("Probabilidades a priori estimadas:")
    for word in words_to_check:
        prob = estimator.get_prior_probability(word)  # Obtener la probabilidad de cada palabra
        print(f"P('{word}') = {prob:.4f}")  # Imprimir la probabilidad con 4 decimales
    
    # Mostrar palabras más probables
    print("\nPalabras más frecuentes:")
    for word, prob in estimator.get_most_probable_words(3):  # Obtener las 3 palabras más probables
        print(f"{word}: {prob:.2%}")  # Imprimir la palabra y su probabilidad en porcentaje