import itertools
from collections import defaultdict

class MedicalDiagnosisSystem:
    def __init__(self):
        # Variables y sus posibles valores
        # Representa las variables del modelo y los valores que pueden tomar
        self.variables = {
            'Enfermedad': ['Gripe', 'Resfriado', 'Neumonia'],
            'Fiebre': ['Alta', 'Moderada', 'Ninguna'],
            'Tos': ['Presente', 'Ausente'],
            'DolorCabeza': ['Presente', 'Ausente']
        }
        
        # Probabilidades a priori
        # Probabilidad inicial de cada enfermedad sin considerar síntomas
        self.priors = {
            'Enfermedad': {'Gripe': 0.3, 'Resfriado': 0.5, 'Neumonia': 0.2}
        }
        
        # Probabilidades condicionales
        # Probabilidad de cada síntoma dado que el paciente tiene una enfermedad específica
        self.conditionals = {
            'Fiebre': {
                'Gripe': {'Alta': 0.6, 'Moderada': 0.3, 'Ninguna': 0.1},
                'Resfriado': {'Alta': 0.1, 'Moderada': 0.4, 'Ninguna': 0.5},
                'Neumonia': {'Alta': 0.8, 'Moderada': 0.15, 'Ninguna': 0.05}
            },
            'Tos': {
                'Gripe': {'Presente': 0.8, 'Ausente': 0.2},
                'Resfriado': {'Presente': 0.6, 'Ausente': 0.4},
                'Neumonia': {'Presente': 0.95, 'Ausente': 0.05}
            },
            'DolorCabeza': {
                'Gripe': {'Presente': 0.7, 'Ausente': 0.3},
                'Resfriado': {'Presente': 0.5, 'Ausente': 0.5},
                'Neumonia': {'Presente': 0.3, 'Ausente': 0.7}
            }
        }
    
    def joint_probability(self, assignment):
        """
        Calcula la probabilidad conjunta P(Enfermedad, Fiebre, Tos, DolorCabeza)
        usando la regla de la cadena.
        
        Args:
            assignment: Diccionario que asigna valores a todas las variables (ej. {'Enfermedad': 'Gripe', 'Fiebre': 'Alta', ...})
        
        Returns:
            Probabilidad conjunta de la asignación dada.
        """
        # Obtener la probabilidad a priori de la enfermedad
        prob = self.priors['Enfermedad'][assignment['Enfermedad']]
        
        # Multiplicar por las probabilidades condicionales de los síntomas
        for symptom in ['Fiebre', 'Tos', 'DolorCabeza']:
            prob *= self.conditionals[symptom][assignment['Enfermedad']][assignment[symptom]]
        
        return prob
    
    def enumerate_ask(self, query_var, evidence):
        """
        Realiza inferencia por enumeración para calcular P(query_var | evidence).
        
        Args:
            query_var: Variable de consulta (ej. 'Enfermedad').
            evidence: Diccionario con las evidencias observadas (ej. {'Fiebre': 'Alta', 'Tos': 'Presente'}).
        
        Returns:
            Distribución de probabilidad condicional P(query_var | evidence).
        """
        # Identificar las variables ocultas (no son query_var ni están en evidence)
        hidden_vars = [var for var in self.variables if var != query_var and var not in evidence]
        
        # Inicializar la distribución de probabilidad para la variable de consulta
        distribution = defaultdict(float)
        
        # Generar todas las combinaciones posibles de valores para las variables ocultas
        value_combinations = itertools.product(
            *[self.variables[var] for var in hidden_vars]
        )
        
        # Iterar sobre todas las combinaciones de valores de las variables ocultas
        for combo in value_combinations:
            # Crear una asignación completa con las evidencias y la combinación actual
            assignment = evidence.copy()
            assignment.update(zip(hidden_vars, combo))
            
            # Iterar sobre todos los valores posibles de la variable de consulta
            for value in self.variables[query_var]:
                # Crear una asignación completa incluyendo el valor de la variable de consulta
                full_assignment = assignment.copy()
                full_assignment[query_var] = value
                
                # Calcular la probabilidad conjunta para esta asignación
                prob = self.joint_probability(full_assignment)
                
                # Sumar la probabilidad al valor correspondiente en la distribución
                distribution[value] += prob
        
        # Normalizar la distribución para que las probabilidades sumen 1
        total = sum(distribution.values())
        if total > 0:
            for key in distribution:
                distribution[key] /= total
        
        return dict(distribution)
    
    def diagnose(self, symptoms):
        """
        Realiza un diagnóstico basado en los síntomas observados.
        
        Args:
            symptoms: Diccionario con los síntomas observados (ej. {'Fiebre': 'Alta', 'Tos': 'Presente'}).
        
        Returns:
            Distribución de probabilidad de cada enfermedad dado los síntomas.
        """
        return self.enumerate_ask('Enfermedad', symptoms)

# Ejemplo de uso
if __name__ == "__main__":
    # Crear una instancia del sistema de diagnóstico
    system = MedicalDiagnosisSystem()
    
    # Caso 1: Paciente con fiebre alta y tos
    symptoms_1 = {'Fiebre': 'Alta', 'Tos': 'Presente'}
    diagnosis_1 = system.diagnose(symptoms_1)
    print("\nDiagnóstico para fiebre alta y tos:")
    for disease, prob in diagnosis_1.items():
        print(f"{disease}: {prob:.2%}")
    
    # Caso 2: Paciente con fiebre moderada y dolor de cabeza
    symptoms_2 = {'Fiebre': 'Moderada', 'DolorCabeza': 'Presente'}
    diagnosis_2 = system.diagnose(symptoms_2)
    print("\nDiagnóstico para fiebre moderada y dolor de cabeza:")
    for disease, prob in diagnosis_2.items():
        print(f"{disease}: {prob:.2%}")
    
    # Caso 3: Paciente con todos los síntomas
    symptoms_3 = {'Fiebre': 'Alta', 'Tos': 'Presente', 'DolorCabeza': 'Presente'}
    diagnosis_3 = system.diagnose(symptoms_3)
    print("\nDiagnóstico para fiebre alta, tos y dolor de cabeza:")
    for disease, prob in diagnosis_3.items():
        print(f"{disease}: {prob:.2%}")