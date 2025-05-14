import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos numéricos.
                    # Es útil para:
                    # - Calcular logaritmos de probabilidades (np.log) para evitar underflow en multiplicaciones.
                    # - Realizar cálculos eficientes con operaciones vectorizadas.

from collections import defaultdict  # Proporciona un diccionario con valores por defecto.
                                     # Es útil para:
                                     # - Contar la frecuencia de palabras en cada clase ("spam" y "no_spam").
                                     # - Evitar errores al acceder a claves inexistentes, ya que devuelve un valor predeterminado (en este caso, 0).

class NaiveBayesSpamFilter:
    def __init__(self):
        # Inicialización de las probabilidades a priori para las clases "spam" y "no_spam"
        self.priors = {"spam": 0.5, "no_spam": 0.5}
        
        # Diccionarios para contar la frecuencia de palabras en cada clase
        self.word_counts = {"spam": defaultdict(int), "no_spam": defaultdict(int)}
        
        # Totales de palabras en cada clase
        self.class_word_totals = {"spam": 0, "no_spam": 0}
        
        # Conjunto para almacenar el vocabulario único (todas las palabras vistas)
        self.vocabulary = set()

    def train(self, emails, labels):
        """Entrena el modelo con una lista de emails y sus etiquetas correspondientes"""
        for email, label in zip(emails, labels):
            # Divide el email en palabras y actualiza los contadores
            for word in email.split():
                self.word_counts[label][word] += 1  # Incrementa el conteo de la palabra para la clase
                self.class_word_totals[label] += 1  # Incrementa el total de palabras para la clase
                self.vocabulary.add(word)  # Añade la palabra al vocabulario
        
        # Calcula el tamaño del vocabulario (para suavizado de Laplace)
        self.vocab_size = len(self.vocabulary)

    def predict(self, email):
        """Clasifica un nuevo email como 'spam' o 'no_spam'"""
        # Inicializa los puntajes logarítmicos con las probabilidades a priori
        spam_score = np.log(self.priors["spam"])
        no_spam_score = np.log(self.priors["no_spam"])
        
        # Procesa cada palabra del email
        for word in email.split():
            if word in self.vocabulary:  # Solo considera palabras que están en el vocabulario
                # Calcula la probabilidad condicional con suavizado de Laplace
                p_word_spam = (self.word_counts["spam"].get(word, 0) + 1) / \
                              (self.class_word_totals["spam"] + self.vocab_size)
                p_word_nospam = (self.word_counts["no_spam"].get(word, 0) + 1) / \
                                (self.class_word_totals["no_spam"] + self.vocab_size)
                
                # Suma los logaritmos de las probabilidades condicionales
                spam_score += np.log(p_word_spam)
                no_spam_score += np.log(p_word_nospam)
        
        # Devuelve la clase con el puntaje más alto
        return "spam" if spam_score > no_spam_score else "no_spam"

# Ejemplo de uso
emails = [
    "oferta ganador gratis premio",  # Email etiquetado como spam
    "reunión proyecto equipo trabajo",  # Email etiquetado como no_spam
    "compra venta oferta exclusiva",  # Email etiquetado como spam
    "presentación informe reunión"  # Email etiquetado como no_spam
]
labels = ["spam", "no_spam", "spam", "no_spam"]  # Etiquetas correspondientes

# Crea una instancia del filtro Naive Bayes y entrena el modelo
nb = NaiveBayesSpamFilter()
nb.train(emails, labels)

# Prueba con un nuevo email
test_email = "oferta reunión"  # Email a clasificar
print(f"'{test_email}' es clasificado como: {nb.predict(test_email)}")