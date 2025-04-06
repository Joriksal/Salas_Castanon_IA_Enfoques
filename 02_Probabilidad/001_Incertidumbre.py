import numpy as np  # Importación de la biblioteca numpy (aunque no se utiliza en este código)

class DecisionSystem:
    def __init__(self):
        """
        Inicializa el sistema de decisión con las probabilidades iniciales (priors)
        y las probabilidades condicionales de los síntomas dados las enfermedades (likelihoods).
        """
        # Probabilidades iniciales de cada enfermedad
        self.disease_prob = {
            'Gripe': 0.30,      # Probabilidad inicial de tener gripe
            'Resfriado': 0.50,  # Probabilidad inicial de tener resfriado
            'Neumonía': 0.20    # Probabilidad inicial de tener neumonía
        }
        
        # Probabilidades condicionales de los síntomas dados cada enfermedad
        self.symptom_given_disease = {
            'Gripe': {'Fiebre': 0.90, 'Tos': 0.70, 'Dolor cabeza': 0.80},
            'Resfriado': {'Fiebre': 0.10, 'Tos': 0.60, 'Dolor cabeza': 0.50},
            'Neumonía': {'Fiebre': 0.95, 'Tos': 0.90, 'Dolor cabeza': 0.40}
        }
    
    def update_probabilities(self, observed_symptoms):
        """
        Actualiza las probabilidades de las enfermedades dado los síntomas observados
        usando el teorema de Bayes.
        
        Parámetros:
        - observed_symptoms: Diccionario donde las claves son los síntomas y los valores
          son True (presente) o False (ausente).
        
        Retorna:
        - Diccionario con las probabilidades actualizadas de cada enfermedad.
        """
        posterior_probs = {}  # Diccionario para almacenar las probabilidades posteriores
        total_prob = 0.0      # Variable para calcular la suma total de probabilidades

        # Iterar sobre cada enfermedad para calcular su probabilidad posterior
        for disease in self.disease_prob:
            likelihood = 1.0  # Inicializar la probabilidad condicional P(síntomas|enfermedad)
            
            # Calcular el likelihood basado en los síntomas observados
            for symptom, present in observed_symptoms.items():
                if present:
                    # Si el síntoma está presente, multiplicar por P(síntoma|enfermedad)
                    likelihood *= self.symptom_given_disease[disease].get(symptom, 0.01)
                else:
                    # Si el síntoma está ausente, multiplicar por 1 - P(síntoma|enfermedad)
                    likelihood *= (1 - self.symptom_given_disease[disease].get(symptom, 0.01))
            
            # Calcular la probabilidad posterior sin normalizar:
            # P(enfermedad|síntomas) ∝ P(síntomas|enfermedad) * P(enfermedad)
            posterior = likelihood * self.disease_prob[disease]
            posterior_probs[disease] = posterior  # Guardar la probabilidad posterior
            total_prob += posterior  # Sumar al total para normalización

        # Normalizar las probabilidades para que sumen 1
        for disease in posterior_probs:
            posterior_probs[disease] /= total_prob
        
        return posterior_probs  # Retornar las probabilidades normalizadas

# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia del sistema de decisión
    system = DecisionSystem()
    
    # Definir los síntomas observados (True = presente, False = ausente)
    symptoms = {
        'Fiebre': True,       # El paciente tiene fiebre
        'Tos': True,          # El paciente tiene tos
        'Dolor cabeza': False # El paciente no tiene dolor de cabeza
    }
    
    # Calcular las probabilidades actualizadas de las enfermedades
    updated_probs = system.update_probabilities(symptoms)
    
    # Mostrar los resultados de las probabilidades actualizadas
    print("Probabilidades de diagnóstico actualizadas:")
    for disease, prob in updated_probs.items():
        print(f"{disease}: {prob:.2%}")  # Mostrar cada enfermedad con su probabilidad en porcentaje