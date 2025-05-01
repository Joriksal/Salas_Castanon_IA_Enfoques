class KDecisionList:
    def __init__(self, rules):
        """
        Inicializa la lista de decisión con un conjunto de reglas predefinidas.
        Cada regla es una tupla que contiene:
        - Una condición: una función lambda que evalúa atributos de una muestra.
        - Un resultado: el valor que se retorna si la condición es verdadera.
        """
        self.rules = rules

    def predict(self, sample):
        """
        Evalúa una muestra (representada como un diccionario de atributos) 
        en orden secuencial según las reglas definidas.
        
        Parámetros:
        - sample (dict): Diccionario que representa los atributos de la muestra.

        Retorna:
        - El resultado asociado a la primera regla cuya condición sea verdadera.
        - "Rechazado" si ninguna regla aplica.
        """
        for condition, outcome in self.rules:
            # Verifica si la condición de la regla es verdadera para la muestra.
            if condition(sample):
                return outcome  # Retorna el resultado asociado a la regla.
        return "Rechazado"  # Valor por defecto si ninguna regla aplica.

# --- Ejemplo de aplicación: Clasificación para aprobación de préstamos bancarios ---
if __name__ == "__main__":
    # Definir un conjunto de reglas para la aprobación de préstamos.
    # Cada regla tiene una condición (lambda) y un resultado asociado.
    rules = [
        # Regla 1: Si el cliente tiene buen crédito Y altos ingresos -> Aprobado.
        (lambda x: x["credito"] == "bueno" and x["ingresos"] == "altos", "Aprobado"),
        # Regla 2: Si el cliente tiene buen crédito (sin importar los ingresos) -> Aprobado con revisión.
        (lambda x: x["credito"] == "bueno", "Aprobado (con revisión)"),
        # Regla 3: Si el cliente tiene ingresos altos (sin importar el crédito) -> Revisión manual.
        (lambda x: x["ingresos"] == "altos", "Revisión manual"),
    ]

    # Crear una instancia del clasificador basado en la lista de decisión.
    loan_classifier = KDecisionList(rules)

    # Definir casos de prueba (clientes con diferentes atributos).
    clients = [
        {"credito": "bueno", "ingresos": "altos"},  # Caso 1: Aprobado.
        {"credito": "bueno", "ingresos": "bajos"},  # Caso 2: Aprobado (con revisión).
        {"credito": "malo", "ingresos": "altos"},   # Caso 3: Revisión manual.
        {"credito": "malo", "ingresos": "bajos"},   # Caso 4: Rechazado (por defecto).
    ]

    # Clasificar cada cliente según las reglas definidas.
    for client in clients:
        # Obtener la decisión para el cliente actual.
        decision = loan_classifier.predict(client)
        # Imprimir el resultado de la clasificación.
        print(f"Cliente: {client} -> Decisión: {decision}")