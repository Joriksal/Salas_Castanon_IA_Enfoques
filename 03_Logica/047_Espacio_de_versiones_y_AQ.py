import numpy as np

class AQAlgorithm:
    def __init__(self):
        # Inicializa una lista vacía para almacenar las reglas generadas
        self.rules = []

    def fit(self, X, y):
        """
        Genera reglas a partir de los ejemplos positivos y negativos.
        :param X: Matriz de ejemplos (atributos).
        :param y: Vector de etiquetas (1 = positivo, 0 = negativo).
        """
        # Filtra los ejemplos positivos y negativos
        positives = X[y == 1]
        negatives = X[y == 0]

        # Genera reglas para cada ejemplo positivo
        for example in positives:
            rule = self._generate_rule(example, negatives)
            # Agrega la regla si es válida y no está duplicada
            if rule and rule not in self.rules:
                self.rules.append(rule)

    def _generate_rule(self, example, negatives):
        """
        Genera la regla más específica que cubre un ejemplo positivo
        y excluye todos los ejemplos negativos.
        :param example: Ejemplo positivo.
        :param negatives: Lista de ejemplos negativos.
        :return: Regla generada como un diccionario {índice_atributo: valor}.
        """
        rule = {}
        for attr_idx in range(len(example)):
            # Verifica si el valor del atributo no aparece en ningún ejemplo negativo
            if all(example[attr_idx] != neg[attr_idx] for neg in negatives):
                rule[attr_idx] = example[attr_idx]  # Agrega el atributo a la regla
        return rule if rule else None  # Devuelve la regla si tiene condiciones

    def predict(self, X):
        """
        Predice las etiquetas para una matriz de ejemplos.
        :param X: Matriz de ejemplos a predecir.
        :return: Vector de predicciones (1 = positivo, 0 = negativo).
        """
        y_pred = []
        for sample in X:
            pred = 0  # Por defecto, la predicción es negativa
            for rule in self.rules:
                # Verifica si el ejemplo cumple con todas las condiciones de la regla
                if all(sample[idx] == val for idx, val in rule.items()):
                    pred = 1  # Si cumple, la predicción es positiva
                    break
            y_pred.append(pred)
        return np.array(y_pred)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Datos de entrenamiento: [Fiebre, Tos, Dolor], etiqueta (1 = enfermo, 0 = sano)
    X = np.array([
        ["alta", "si", "si"],   # Ejemplo positivo (enfermo)
        ["alta", "no", "si"],   # Ejemplo positivo (enfermo)
        ["baja", "si", "no"],   # Ejemplo negativo (sano)
        ["media", "si", "no"],  # Ejemplo negativo (sano)
    ], dtype=object)

    y = np.array([1, 1, 0, 0])  # Etiquetas correspondientes

    # Inicializa el algoritmo AQ y ajusta las reglas con los datos de entrenamiento
    aq = AQAlgorithm()
    aq.fit(X, y)

    # Nombres de los atributos para facilitar la interpretación de las reglas
    attr_names = ["Fiebre", "Tos", "Dolor"]
    print("Reglas generadas:")
    for i, rule in enumerate(aq.rules, 1):
        # Convierte las reglas en una representación legible
        conds = [f"{attr_names[idx]}={val}" for idx, val in rule.items()]
        print(f"Regla {i}: SI {' Y '.join(conds)} ENTONCES Enfermo")

    # Casos de prueba para evaluar las predicciones del modelo
    test_cases = np.array([
        ["alta", "si", "no"],  # No cubierto completamente → Predicción: 0
        ["baja", "no", "si"],  # No cubierto → Predicción: 0
        ["alta", "no", "si"],  # Igual que un ejemplo positivo → Predicción: 1
    ], dtype=object)

    # Realiza las predicciones para los casos de prueba
    preds = aq.predict(test_cases)
    print("\nPredicciones:", preds)
