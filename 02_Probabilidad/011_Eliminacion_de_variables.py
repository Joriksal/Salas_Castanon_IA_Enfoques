import numpy as np  # Librería para operaciones matemáticas y manejo de arreglos numéricos.
                    # Es útil para:
                    # - Representar y manipular probabilidades como arreglos (vectores y matrices).
                    # - Realizar cálculos eficientes con operaciones vectorizadas.
                    # - Manejar estructuras de datos como matrices de probabilidades condicionales.

# Probabilidades a priori de los componentes C1, C2 y C3
# Estas representan la probabilidad de que cada componente esté en estado 0 (funcionando) o 1 (fallando)
P_C1 = np.array([0.95, 0.05])  # P(C1=0), P(C1=1)
P_C2 = np.array([0.90, 0.10])  # P(C2=0), P(C2=1)
P_C3 = np.array([0.85, 0.15])  # P(C3=0), P(C3=1)

# Probabilidades condicionales de los sensores S1 y S2 dado el estado de los componentes C1 y C2
# Estas matrices indican cómo los sensores detectan fallas en los componentes
P_S1_C1 = np.array([[0.90, 0.10], [0.10, 0.90]])  # P(S1|C1)
P_S2_C2 = np.array([[0.80, 0.20], [0.05, 0.95]])  # P(S2|C2)

# Función que define la probabilidad de falla crítica (F) dado el estado de C3, S1 y S2
def P_F_given_C3_S1_S2(C3, S1, S2):
    """
    Calcula la probabilidad de falla crítica (F) dado el estado de C3, S1 y S2.
    """
    if C3 == 1:  # Si el componente C3 falla
        return 0.95  # Alta probabilidad de falla crítica
    elif S1 == 1 or S2 == 1:  # Si alguno de los sensores detecta una falla
        return 0.70  # Probabilidad moderada de falla crítica
    else:  # Si no hay fallas detectadas
        return 0.01  # Baja probabilidad de falla crítica

# Función principal para calcular la probabilidad de falla crítica P(F|S1=1, S2=0)
def calcular_prob_falla_critica():
    """
    Calcula la probabilidad de falla crítica (F) dado que S1=1 y S2=0
    utilizando el método de eliminación de variables.
    """
    # Inicializamos la probabilidad marginal de F
    P_F = 0.0

    # Iteramos sobre todas las combinaciones posibles de los estados de C1, C2 y C3
    for C1_val in [0, 1]:  # Estado de C1 (0 o 1)
        for C2_val in [0, 1]:  # Estado de C2 (0 o 1)
            for C3_val in [0, 1]:  # Estado de C3 (0 o 1)
                # Calculamos P(F=1 | C3, S1=1, S2=0)
                prob_F = P_F_given_C3_S1_S2(C3_val, 1, 0)

                # Calculamos P(S1=1|C1) y P(S2=0|C2) a partir de las matrices condicionales
                prob_S1_given_C1 = P_S1_C1[C1_val, 1]  # Probabilidad de que S1=1 dado C1
                prob_S2_given_C2 = P_S2_C2[C2_val, 0]  # Probabilidad de que S2=0 dado C2

                # Calculamos la probabilidad conjunta de C1, C2, C3, S1 y S2
                joint_prob = (
                    P_C1[C1_val] * P_C2[C2_val] * P_C3[C3_val] *  # Probabilidades a priori
                    prob_S1_given_C1 * prob_S2_given_C2 *          # Probabilidades condicionales
                    prob_F                                         # Probabilidad de F dado C3, S1, S2
                )

                # Sumamos la probabilidad conjunta al total de P(F)
                P_F += joint_prob

    # Normalizamos la probabilidad para obtener P(F=1 | S1=1, S2=0)
    # Esto asegura que las probabilidades sumen 1
    P_F_normalized = P_F / (P_F + (1 - P_F))  # Asumiendo simetría para P(F=0)
    
    return P_F_normalized

# Resultado final
# Calculamos la probabilidad de falla crítica dado que S1=1 y S2=0
prob_falla = calcular_prob_falla_critica()
print(f"Probabilidad de falla crítica dado S1=1 y S2=0: {prob_falla:.4f}")