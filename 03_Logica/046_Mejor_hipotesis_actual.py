def find_s(training_data):
    """
    Algoritmo Find-S para encontrar la Mejor Hipótesis Actual (MHA).
    - training_data: Lista de tuplas (atributos, etiqueta), donde etiqueta es True (positivo) o False (negativo).
    - Devuelve la hipótesis más específica que cubra todos los ejemplos positivos.
    """
    hypothesis = None  # Hipótesis inicial vacía (sin restricciones)

    # Iterar sobre los datos de entrenamiento
    for attributes, label in training_data:
        if label:  # Solo se consideran los ejemplos positivos
            if hypothesis is None:
                # Si no hay hipótesis inicial, se toma el primer ejemplo positivo como base
                hypothesis = list(attributes)
            else:
                # Comparar cada atributo de la hipótesis con el ejemplo actual
                for i in range(len(hypothesis)):
                    if hypothesis[i] != attributes[i]:
                        # Si hay una diferencia, generalizar el atributo con '?'
                        hypothesis[i] = '?'

    return hypothesis  # Retornar la hipótesis más específica encontrada


if __name__ == "__main__":
    # Datos de entrenamiento: (edad, ingresos, historial_crediticio), etiqueta (True = préstamo aprobado)
    training_data = [
        (("joven", "alto", "bueno"), True),   # Ejemplo positivo
        (("joven", "medio", "bueno"), True), # Ejemplo positivo
        (("adulto", "bajo", "malo"), False), # Ejemplo negativo
        (("adulto", "medio", "bueno"), True),# Ejemplo positivo
    ]

    # Encontrar la mejor hipótesis actual (MHA) usando el algoritmo Find-S
    best_hypothesis = find_s(training_data)
    print(f"Mejor Hipótesis Actual (MHA): {best_hypothesis}")

    # Función de predicción basada en la hipótesis encontrada
    def predict(hypothesis, sample):
        """
        Predice si un ejemplo cumple con la hipótesis.
        - hypothesis: Hipótesis generada por Find-S.
        - sample: Ejemplo a evaluar (tupla de atributos).
        - Devuelve True si el ejemplo cumple con la hipótesis, False en caso contrario.
        """
        # Verificar que cada atributo del ejemplo cumpla con la hipótesis
        return all(h == '?' or h == s for h, s in zip(hypothesis, sample))

    # Prueba con un nuevo cliente
    new_client = ("joven", "bajo", "bueno")  # Atributos del nuevo cliente
    aprobado = predict(best_hypothesis, new_client)  # Evaluar si cumple con la hipótesis
    print(f"Cliente {new_client} -> Préstamo aprobado?: {'Sí' if aprobado else 'No'}")
