import numpy as np
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

# 1. Definir la estructura de la red bayesiana
# Cada tupla ('Nodo1', 'Nodo2') indica que Nodo1 es un padre de Nodo2.
model = DiscreteBayesianNetwork([
    ('L1', 'L2'),  # L1 influye en L2
    ('L1', 'L3'),  # L1 influye en L3
    ('L2', 'L4')   # L2 influye en L4
])

# 2. Definir las Tablas de Probabilidad Condicional (CPDs)

# CPD para L1: Probabilidades iniciales de los estados de L1 (sin padres)
# L1 tiene 3 estados posibles: +, -, →
cpd_l1 = TabularCPD('L1', 3, [[0.33], [0.33], [0.33]])  # Probabilidades uniformes

# CPD para L2: Probabilidades condicionales de L2 dado L1
cpd_l2 = TabularCPD(
    variable='L2',          # Nodo actual
    variable_card=3,        # Número de estados posibles de L2
    values=[                # Matriz de probabilidades condicionales P(L2|L1)
        [0.8, 0.1, 0.1],    # P(L2=+|L1=+)
        [0.1, 0.8, 0.1],    # P(L2=-|L1=-)
        [0.1, 0.1, 0.8]     # P(L2=→|L1=→)
    ],
    evidence=['L1'],        # Nodo padre
    evidence_card=[3]       # Número de estados posibles del nodo padre (L1)
)

# CPD para L3: Probabilidades condicionales de L3 dado L1
cpd_l3 = TabularCPD(
    variable='L3',          # Nodo actual
    variable_card=3,        # Número de estados posibles de L3
    values=[                # Matriz de probabilidades condicionales P(L3|L1)
        [0.7, 0.2, 0.1],    # P(L3=+|L1=+)
        [0.2, 0.7, 0.1],    # P(L3=-|L1=-)
        [0.1, 0.1, 0.8]     # P(L3=→|L1=→)
    ],
    evidence=['L1'],        # Nodo padre
    evidence_card=[3]       # Número de estados posibles del nodo padre (L1)
)

# CPD para L4: Probabilidades condicionales de L4 dado L2
cpd_l4 = TabularCPD(
    variable='L4',          # Nodo actual
    variable_card=3,        # Número de estados posibles de L4
    values=[                # Matriz de probabilidades condicionales P(L4|L2)
        [0.7, 0.2, 0.1],    # P(L4=+|L2=+)
        [0.2, 0.7, 0.1],    # P(L4=-|L2=-)
        [0.1, 0.1, 0.8]     # P(L4=→|L2=→)
    ],
    evidence=['L2'],        # Nodo padre
    evidence_card=[3]       # Número de estados posibles del nodo padre (L2)
)

# 3. Añadir todas las CPDs al modelo
# Esto asocia las probabilidades definidas con la estructura de la red
model.add_cpds(cpd_l1, cpd_l2, cpd_l3, cpd_l4)

# 4. Verificar que el modelo es válido
# Comprueba que todas las CPDs están correctamente definidas y que el modelo es coherente
model.check_model()

# 5. Realizar inferencia en la red bayesiana
# Usamos VariableElimination para calcular probabilidades posteriores
inference = VariableElimination(model)

# Consultar la probabilidad de L4 dado que L1 está en el estado '+'
result = inference.query(variables=['L4'], evidence={'L1': 0})  # L1=0 corresponde a '+'
print(result)  # Imprime las probabilidades de los estados de L4