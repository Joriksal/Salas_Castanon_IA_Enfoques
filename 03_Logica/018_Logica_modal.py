# Clase que representa un modelo de Kripke, utilizado en lógica modal
# Un modelo de Kripke es una estructura matemática que se utiliza para evaluar fórmulas de lógica modal.
class ModeloKripke:
    def __init__(self):
        # Inicializa el conjunto de mundos posibles en el modelo
        self.mundos = set()
        # Diccionario que representa las relaciones de accesibilidad entre mundos para cada agente
        # Formato: {agente: {origen: {destino}}}
        self.relaciones = {}
        # Diccionario que almacena las valuaciones de proposiciones en cada mundo
        # Formato: {mundo: {proposición: valor_de_verdad}}
        self.valuaciones = {}

    # Método para agregar un mundo al modelo
    def agregar_mundo(self, mundo: str):
        # Añade el mundo al conjunto de mundos posibles
        self.mundos.add(mundo)
        # Inicializa las valuaciones para el mundo si aún no existen
        if mundo not in self.valuaciones:
            self.valuaciones[mundo] = {}

    # Método para definir una relación de accesibilidad entre dos mundos para un agente
    def agregar_relacion(self, agente: str, origen: str, destino: str):
        # Si el agente no tiene relaciones definidas, inicializa su entrada
        if agente not in self.relaciones:
            self.relaciones[agente] = {}
        # Si el mundo de origen no tiene destinos definidos, inicializa su entrada
        if origen not in self.relaciones[agente]:
            self.relaciones[agente][origen] = set()
        # Agrega el mundo de destino a las relaciones de accesibilidad del agente
        self.relaciones[agente][origen].add(destino)
        # Asegura que los mundos involucrados existan en el modelo
        self.agregar_mundo(origen)
        self.agregar_mundo(destino)

    # Método para asignar un valor de verdad a una proposición en un mundo específico
    def asignar_valuacion(self, mundo: str, proposicion: str, valor: bool):
        # Si el mundo no existe en las valuaciones, lo agrega
        if mundo not in self.valuaciones:
            self.agregar_mundo(mundo)
        # Asigna el valor de verdad a la proposición en el mundo dado
        self.valuaciones[mundo][proposicion] = valor


# Clase que evalúa fórmulas modales en un modelo de Kripke
class EvaluadorModal:
    def __init__(self, modelo: ModeloKripke):
        # Recibe una instancia de ModeloKripke para realizar las evaluaciones
        self.modelo = modelo

    # Método para evaluar una fórmula en un mundo específico
    def evaluar(self, formula: str, mundo: str) -> bool:
        # Caso: fórmula modal □ (necesidad)
        if formula.startswith('□_'):
            # Extrae el agente y la subfórmula de la fórmula modal
            agente, subformula = formula[2:].split(' ', 1)
            # Verifica si la subfórmula es verdadera en todos los mundos accesibles desde el mundo actual
            return all(
                self.evaluar(subformula, w)
                for w in self.modelo.relaciones.get(agente, {}).get(mundo, set())
            )
        # Caso: fórmula modal ◇ (posibilidad)
        elif formula.startswith('◇_'):
            # Extrae el agente y la subfórmula de la fórmula modal
            agente, subformula = formula[2:].split(' ', 1)
            # Verifica si la subfórmula es verdadera en al menos un mundo accesible desde el mundo actual
            return any(
                self.evaluar(subformula, w)
                for w in self.modelo.relaciones.get(agente, {}).get(mundo, set())
            )
        # Caso: negación (¬)
        elif formula.startswith('¬'):
            # Evalúa la fórmula negada y devuelve el valor contrario
            return not self.evaluar(formula[1:], mundo)
        # Caso: conjunción (∧)
        elif '∧' in formula:
            # Divide la fórmula en las dos partes de la conjunción
            partes = formula.split(' ∧ ', 1)
            # Evalúa ambas partes y devuelve True si ambas son verdaderas
            return self.evaluar(partes[0], mundo) and self.evaluar(partes[1], mundo)
        # Caso: disyunción (∨)
        elif '∨' in formula:
            # Divide la fórmula en las dos partes de la disyunción
            partes = formula.split(' ∨ ', 1)
            # Evalúa ambas partes y devuelve True si al menos una es verdadera
            return self.evaluar(partes[0], mundo) or self.evaluar(partes[1], mundo)
        # Caso: implicación (→)
        elif '→' in formula:
            # Divide la fórmula en las dos partes de la implicación
            partes = formula.split(' → ', 1)
            # Evalúa la implicación lógica: si la primera parte es falsa o la segunda es verdadera
            return (not self.evaluar(partes[0], mundo)) or self.evaluar(partes[1], mundo)
        # Caso: proposición atómica
        else:
            # Devuelve el valor de verdad de la proposición en el mundo actual
            return self.modelo.valuaciones.get(mundo, {}).get(formula, False)


# Función para construir un modelo de conocimiento multiagente
def construir_modelo_conocimiento():
    # Crea una instancia del modelo de Kripke
    modelo = ModeloKripke()
    
    # Define los mundos posibles en el modelo
    mundos = ['w1', 'w2', 'w3']
    for w in mundos:
        modelo.agregar_mundo(w)
    
    # Define las relaciones de accesibilidad entre mundos para los agentes
    modelo.agregar_relacion('Alice', 'w1', 'w2')  # Alice puede acceder de w1 a w2
    modelo.agregar_relacion('Alice', 'w2', 'w1')  # Alice puede acceder de w2 a w1
    modelo.agregar_relacion('Bob', 'w1', 'w3')    # Bob puede acceder de w1 a w3
    
    # Asigna valores de verdad a proposiciones en cada mundo
    modelo.asignar_valuacion('w1', 'p', True)     # En w1, p es verdadero
    modelo.asignar_valuacion('w1', 'q', False)    # En w1, q es falso
    modelo.asignar_valuacion('w2', 'p', False)    # En w2, p es falso
    modelo.asignar_valuacion('w2', 'q', True)     # En w2, q es verdadero
    modelo.asignar_valuacion('w3', 'p', True)     # En w3, p es verdadero
    modelo.asignar_valuacion('w3', 'q', True)     # En w3, q es verdadero
    
    # Devuelve el modelo construido
    return modelo


# Punto de entrada principal del programa
if __name__ == "__main__":
    # Imprime un mensaje inicial
    print("=== Sistema de Lógica Modal ===")
    
    # Construye el modelo de conocimiento
    modelo = construir_modelo_conocimiento()
    # Crea un evaluador modal basado en el modelo
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
        # Evalúa cada fórmula en el mundo w1 y muestra el resultado
        print(f"{f}: {evaluador.evaluar(f, 'w1')}")
    
    # Evaluación de las fórmulas en el mundo w2
    print("\nEvaluación en w2:")
    for f in formulas:
        # Evalúa cada fórmula en el mundo w2 y muestra el resultado
        print(f"{f}: {evaluador.evaluar(f, 'w2')}")