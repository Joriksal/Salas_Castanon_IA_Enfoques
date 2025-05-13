class MedicalDiagnosisSystem:
    def __init__(self):
        """
        Inicializa el sistema de diagnóstico médico con probabilidades a priori
        y probabilidades condicionales para cada enfermedad y síntoma.
        """
        # Probabilidades a priori (prevalencia de enfermedades)
        self.disease_priors = {
            'Gripe': 0.30,      # 30% de probabilidad de tener gripe
            'Resfriado': 0.50,  # 50% de probabilidad de tener resfriado
            'Neumonía': 0.20    # 20% de probabilidad de tener neumonía
        }
        
        # Probabilidades condicionales P(síntoma|enfermedad)
        self.symptom_likelihoods = {
            'Gripe': {
                'Fiebre': 0.90,          # 90% de probabilidad de fiebre si tiene gripe
                'Tos': 0.70,            # 70% de probabilidad de tos si tiene gripe
                'Dolor de cabeza': 0.80, # 80% de probabilidad de dolor de cabeza si tiene gripe
                'Fatiga': 0.60          # 60% de probabilidad de fatiga si tiene gripe
            },
            'Resfriado': {
                'Fiebre': 0.10,         # 10% de probabilidad de fiebre si tiene resfriado
                'Tos': 0.60,            # 60% de probabilidad de tos si tiene resfriado
                'Dolor de cabeza': 0.50, # 50% de probabilidad de dolor de cabeza si tiene resfriado
                'Fatiga': 0.40          # 40% de probabilidad de fatiga si tiene resfriado
            },
            'Neumonía': {
                'Fiebre': 0.95,         # 95% de probabilidad de fiebre si tiene neumonía
                'Tos': 0.90,            # 90% de probabilidad de tos si tiene neumonía
                'Dolor de cabeza': 0.40, # 40% de probabilidad de dolor de cabeza si tiene neumonía
                'Fatiga': 0.70          # 70% de probabilidad de fatiga si tiene neumonía
            }
        }
        
        # Conjunto de todos los síntomas posibles
        self.symptoms = set()
        for disease in self.symptom_likelihoods:
            self.symptoms.update(self.symptom_likelihoods[disease].keys())
    
    def apply_bayes_rule(self, observed_symptoms):
        """
        Aplica la regla de Bayes para calcular las probabilidades posteriores
        de cada enfermedad dado un conjunto de síntomas observados.
        
        Args:
            observed_symptoms (dict): Diccionario de síntomas {síntoma: True/False}
            
        Returns:
            dict: Probabilidades posteriores normalizadas para cada enfermedad.
        """
        posterior_probs = {}  # Diccionario para almacenar las probabilidades posteriores
        total_prob = 0.0      # Suma total de probabilidades para normalización
        
        for disease in self.disease_priors:
            # Inicializar con la probabilidad a priori de la enfermedad
            posterior = self.disease_priors[disease]
            
            # Multiplicar por las probabilidades condicionales de los síntomas observados
            for symptom, present in observed_symptoms.items():
                if present:  # Si el síntoma está presente
                    posterior *= self.symptom_likelihoods[disease].get(symptom, 0.01)
                else:  # Si el síntoma no está presente
                    posterior *= (1 - self.symptom_likelihoods[disease].get(symptom, 0.01))
            
            # Guardar la probabilidad posterior calculada
            posterior_probs[disease] = posterior
            # Sumar al total para normalización
            total_prob += posterior
        
        # Normalizar las probabilidades para que sumen 1
        if total_prob > 0:
            for disease in posterior_probs:
                posterior_probs[disease] /= total_prob
        
        return posterior_probs
    
    def print_diagnosis(self, observed_symptoms):
        """
        Muestra un informe detallado del diagnóstico basado en los síntomas observados.
        
        Args:
            observed_symptoms (dict): Diccionario de síntomas {síntoma: True/False}
            
        Returns:
            dict: Probabilidades posteriores calculadas.
        """
        print("\n=== Sistema de Diagnóstico Médico ===")
        print("Síntomas observados:")
        # Imprimir los síntomas presentes
        for symptom, present in observed_symptoms.items():
            if present:
                print(f"- {symptom}")
        
        print("\nCálculo de probabilidades:")
        # Calcular las probabilidades posteriores usando la regla de Bayes
        posterior_probs = self.apply_bayes_rule(observed_symptoms)
        
        print("\nResultados del diagnóstico:")
        # Imprimir las probabilidades ordenadas de mayor a menor
        for disease, prob in sorted(posterior_probs.items(), key=lambda x: x[1], reverse=True):
            print(f"{disease}: {prob:.2%}")
        
        return posterior_probs

# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia del sistema de diagnóstico
    system = MedicalDiagnosisSystem()
    
    # Caso 1: Diagnóstico para un paciente con síntomas de gripe
    print("\nCaso 1: Paciente con fiebre, tos y dolor de cabeza")
    symptoms_1 = {
        'Fiebre': True,
        'Tos': True,
        'Dolor de cabeza': True,
        'Fatiga': False
    }
    diagnosis_1 = system.print_diagnosis(symptoms_1)
    
    # Caso 2: Diagnóstico para un paciente con síntomas de neumonía
    print("\nCaso 2: Paciente con fiebre alta y tos persistente")
    symptoms_2 = {
        'Fiebre': True,
        'Tos': True,
        'Dolor de cabeza': False,
        'Fatiga': True
    }
    diagnosis_2 = system.print_diagnosis(symptoms_2)
    
    # Caso 3: Diagnóstico para un paciente con síntomas de resfriado
    print("\nCaso 3: Paciente con tos leve sin fiebre")
    symptoms_3 = {
        'Fiebre': False,
        'Tos': True,
        'Dolor de cabeza': False,
        'Fatiga': True
    }
    diagnosis_3 = system.print_diagnosis(symptoms_3)