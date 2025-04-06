
# Importar librerías necesarias
import numpy as np  # Para operaciones matemáticas como logaritmos y exponenciales
from collections import defaultdict  # Para inicializar diccionarios con valores por defecto

# Definición de la clase BayesianFilter
class BayesianFilter:
    def __init__(self):
        """
        Inicializa el filtro bayesiano con estructuras de datos necesarias
        para almacenar conteos y probabilidades.
        """
        # Diccionarios para almacenar conteos de palabras en spam y ham
        self.spam_counts = defaultdict(int)
        self.ham_counts = defaultdict(int)
        
        # Contadores totales de emails spam y ham
        self.total_spam = 0
        self.total_ham = 0
        
        # Probabilidades a priori iniciales (50% para cada clase)
        self.p_spam = 0.5
        self.p_ham = 0.5
        
        # Parámetro de suavizado (Laplace) para evitar probabilidades de cero
        self.alpha = 1
        
        # Conjunto de palabras únicas (vocabulario)
        self.vocab = set()
    
    def train(self, emails):
        """
        Entrena el modelo con una lista de emails etiquetados como spam o ham.
        
        Args:
            emails (list): Lista de tuplas (texto, 'spam'/'ham')
        """
        for text, label in emails:
            # Dividir el texto en palabras y convertir a minúsculas
            words = text.lower().split()
            
            # Si el email es spam
            if label == 'spam':
                for word in words:
                    self.spam_counts[word] += 1  # Incrementar conteo de la palabra en spam
                    self.vocab.add(word)  # Agregar palabra al vocabulario
                self.total_spam += 1  # Incrementar contador total de spam
            else:  # Si el email es ham
                for word in words:
                    self.ham_counts[word] += 1  # Incrementar conteo de la palabra en ham
                    self.vocab.add(word)  # Agregar palabra al vocabulario
                self.total_ham += 1  # Incrementar contador total de ham
        
        # Calcular probabilidades a priori basadas en los conteos totales
        total = self.total_spam + self.total_ham
        self.p_spam = self.total_spam / total
        self.p_ham = self.total_ham / total
    
    def conditional_prob(self, word, label):
        """
        Calcula la probabilidad condicionada P(palabra|clase) usando suavizado de Laplace.
        
        Args:
            word (str): Palabra a evaluar
            label (str): Clase ('spam' o 'ham')
            
        Returns:
            float: Probabilidad condicionada
        """
        if label == 'spam':
            # Obtener conteo de la palabra en spam
            count = self.spam_counts.get(word, 0)
            total = self.total_spam  # Total de emails spam
        else:
            # Obtener conteo de la palabra en ham
            count = self.ham_counts.get(word, 0)
            total = self.total_ham  # Total de emails ham
            
        # Aplicar suavizado de Laplace
        return (count + self.alpha) / (total + self.alpha * len(self.vocab))
    
    def normalize(self, probs):
        """
        Normaliza un diccionario de probabilidades para que sumen 1.
        
        Args:
            probs (dict): Diccionario de probabilidades
            
        Returns:
            dict: Probabilidades normalizadas
        """
        total = sum(probs.values())  # Sumar todas las probabilidades
        return {k: v / total for k, v in probs.items()}  # Dividir cada probabilidad por la suma total
    
    def predict(self, text):
        """
        Predice si un email es spam o ham usando probabilidades condicionadas.
        
        Args:
            text (str): Texto del email
            
        Returns:
            dict: Probabilidades normalizadas para spam y ham
        """
        # Dividir el texto en palabras y convertir a minúsculas
        words = text.lower().split()
        
        # Inicializar con logaritmos de las probabilidades a priori
        p_spam_text = np.log(self.p_spam)
        p_ham_text = np.log(self.p_ham)
        
        # Calcular el producto de las probabilidades condicionadas (en logaritmos)
        for word in words:
            p_spam_text += np.log(self.conditional_prob(word, 'spam'))
            p_ham_text += np.log(self.conditional_prob(word, 'ham'))
        
        # Convertir de logaritmos a probabilidades reales
        raw_probs = {
            'spam': np.exp(p_spam_text),
            'ham': np.exp(p_ham_text)
        }
        
        # Normalizar las probabilidades para que sumen 1
        return self.normalize(raw_probs)

# Ejemplo de uso
if __name__ == "__main__":
    # Datos de entrenamiento: lista de emails con etiquetas ('spam' o 'ham')
    training_emails = [
        ("oferta gratis ganar dinero rápido", "spam"),
        ("reunión proyecto equipo mañana", "ham"),
        ("ganar premio clic aquí", "spam"),
        ("informe mensual adjunto revisión", "ham"),
        ("trabajo desde casa gana 1000 diarios", "spam"),
        ("recordatorio pago factura", "ham")
    ]
    
    # Crear y entrenar el filtro bayesiano
    spam_filter = BayesianFilter()
    spam_filter.train(training_emails)
    
    # Email de prueba para clasificar
    test_email = "oferta de ganar dinero rápido con clic"
    
    # Predecir probabilidades de que el email sea spam o ham
    probabilities = spam_filter.predict(test_email)
    
    # Mostrar resultados de las probabilidades normalizadas
    print("Probabilidades condicionadas normalizadas:")
    print(f"SPAM: {probabilities['spam']:.2%}")  # Probabilidad de ser spam
    print(f"HAM: {probabilities['ham']:.2%}")  # Probabilidad de ser ham