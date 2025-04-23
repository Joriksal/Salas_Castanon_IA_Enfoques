from typing import Dict, List, Set, Tuple, Callable

# Clase que representa un modelo de Kripke, utilizado en lógica modal
class ModeloKripke:
    def __init__(self):
        # Conjunto de mundos posibles
        self.mundos = set()
        # Relaciones de accesibilidad entre mundos para cada agente
        self.relaciones = {}  # {agente: {origen: {destino}}}
        # Valuaciones de proposiciones en cada mundo
        self.valuaciones = {}  # {mundo: {proposición: bool}}

    # Agrega un mundo al modelo
    def agregar_mundo(self, mundo: str):
        self.mundos.add(mundo)
        if mundo not in self.valuaciones:
            self.valuaciones[mundo] = {}

    # Define una relación de accesibilidad entre dos mundos para un agente
    def agregar_relacion(self, agente: str, origen: str, destino: str):
        if agente not in self.relaciones:
            self.relaciones[agente] = {}
        if origen not in self.relaciones[agente]:
            self.relaciones[agente][origen] = set()
        self.relaciones[agente][origen].add(destino)
        # Asegura que los mundos involucrados existan en el modelo
        self.agregar_mundo(origen)
        self.agregar_mundo(destino)

    # Asigna un valor de verdad a una proposición en un mundo específico
    def asignar_valuacion(self, mundo: str, proposicion: str, valor: bool):
        if mundo not in self.valuaciones:
            self.agregar_mundo(mundo)
        self.valuaciones[mundo][proposicion] = valor

# Clase que evalúa fórmulas modales en un modelo de Kripke
class EvaluadorModal:
    def __init__(self, modelo: ModeloKripke):
        self.modelo = modelo

    # Evalúa una fórmula en un mundo específico
    def evaluar(self, formula: str, mundo: str) -> bool:
        # Caso: fórmula modal □ (necesidad)
        if formula.startswith('□_'):
            agente, subformula = formula[2:].split(' ', 1)
            # Verifica si la subfórmula es verdadera en todos los mundos accesibles
            return all(
                self.evaluar(subformula, w)
                for w in self.modelo.relaciones.get(agente, {}).get(mundo, set())
            )
        # Caso: fórmula modal ◇ (posibilidad)
        elif formula.startswith('◇_'):
            agente, subformula = formula[2:].split(' ', 1)
            # Verifica si la subfórmula es verdadera en al menos un mundo accesible
            return any(
                self.evaluar(subformula, w)
                for w in self.modelo.relaciones.get(agente, {}).get(mundo, set())
            )
        # Caso: negación
        elif formula.startswith('¬'):
            return not self.evaluar(formula[1:], mundo)
        # Caso: conjunción
        elif '∧' in formula:
            partes = formula.split(' ∧ ', 1)
            return self.evaluar(partes[0], mundo) and self.evaluar(partes[1], mundo)
        # Caso: disyunción
        elif '∨' in formula:
            partes = formula.split(' ∨ ', 1)
            return self.evaluar(partes[0], mundo) or self.evaluar(partes[1], mundo)
        # Caso: implicación
        elif '→' in formula:
            partes = formula.split(' → ', 1)
            return (not self.evaluar(partes[0], mundo)) or self.evaluar(partes[1], mundo)
        # Caso: proposición atómica
        else:
            return self.modelo.valuaciones.get(mundo, {}).get(formula, False)

# Función para construir un modelo de conocimiento multiagente
def construir_modelo_conocimiento():
    modelo = ModeloKripke()
    
    # Define los mundos posibles
    mundos = ['w1', 'w2', 'w3']
    for w in mundos:
        modelo.agregar_mundo(w)
    
    # Define las relaciones de accesibilidad entre mundos para los agentes
    modelo.agregar_relacion('Alice', 'w1', 'w2')
    modelo.agregar_relacion('Alice', 'w2', 'w1')
    modelo.agregar_relacion('Bob', 'w1', 'w3')
    
    # Asigna valores de verdad a proposiciones en cada mundo
    modelo.asignar_valuacion('w1', 'p', True)
    modelo.asignar_valuacion('w1', 'q', False)
    modelo.asignar_valuacion('w2', 'p', False)
    modelo.asignar_valuacion('w2', 'q', True)
    modelo.asignar_valuacion('w3', 'p', True)
    modelo.asignar_valuacion('w3', 'q', True)
    
    return modelo

# Punto de entrada principal
if __name__ == "__main__":
    print("=== Sistema de Lógica Modal ===")
    modelo = construir_modelo_conocimiento()
    evaluador = EvaluadorModal(modelo)
    
    # Lista de fórmulas a evaluar
    formulas = [
        'p',                # Proposición atómica
        '¬p',               # Negación
        '□_Alice p',        # Necesidad modal para Alice
        '◇_Alice q',        # Posibilidad modal para Alice
        '□_Bob (p ∨ q)',    # Necesidad modal para Bob con disyunción
        'p → □_Alice ¬q'    # Implicación con necesidad modal
    ]
    
    # Evaluación de las fórmulas en el mundo w1
    print("\nEvaluación en w1:")
    for f in formulas:
        print(f"{f}: {evaluador.evaluar(f, 'w1')}")
    
    # Evaluación de las fórmulas en el mundo w2
    print("\nEvaluación en w2:")
    for f in formulas:
        print(f"{f}: {evaluador.evaluar(f, 'w2')}")