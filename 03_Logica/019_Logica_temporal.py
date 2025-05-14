# Importamos las librerías necesarias
from enum import Enum, auto  # Para definir enumeraciones que representan operadores temporales
from typing import List, Dict, Union  # Para definir tipos de datos más claros y específicos

# Enumeración para representar los operadores temporales de Lógica Temporal Lineal (LTL)
class TemporalOperator(Enum):
    GLOBALLY = auto()  # Operador G (siempre): algo es verdadero en todos los estados futuros
    FINALLY = auto()   # Operador F (eventualmente): algo será verdadero en algún estado futuro
    NEXT = auto()      # Operador X (siguiente): algo es verdadero en el siguiente estado
    UNTIL = auto()     # Operador U (hasta): algo es verdadero hasta que otra cosa sea verdadera

# Clase que representa una fórmula temporal
class TemporalFormula:
    def __init__(self, operator: TemporalOperator, *subformulas):
        """
        Constructor para inicializar una fórmula temporal.
        :param operator: Operador temporal (G, F, X, U).
        :param subformulas: Subfórmulas asociadas al operador.
        """
        self.operator = operator  # El operador temporal de la fórmula
        self.subformulas = subformulas  # Las subfórmulas que dependen del operador
    
    def __str__(self):
        """
        Representación en texto de la fórmula temporal.
        :return: Cadena que representa la fórmula en notación LTL.
        """
        # Mapeo de operadores a sus representaciones en texto
        op_map = {
            TemporalOperator.GLOBALLY: 'G',
            TemporalOperator.FINALLY: 'F',
            TemporalOperator.NEXT: 'X',
            TemporalOperator.UNTIL: 'U'
        }
        # Formateo de la fórmula dependiendo del operador
        if self.operator == TemporalOperator.UNTIL:
            # Para el operador U (hasta), se necesitan dos subfórmulas
            return f"({self.subformulas[0]} U {self.subformulas[1]})"
        # Para los operadores G, F y X, solo se necesita una subfórmula
        return f"{op_map[self.operator]}({self.subformulas[0]})"

# Clase que verifica fórmulas temporales sobre un rastro de ejecución
class TemporalVerifier:
    def evaluate(self, formula: Union[str, TemporalFormula], trace: List[Dict[str, bool]], position: int = 0) -> bool:
        """
        Evalúa una fórmula temporal en un rastro de ejecución desde una posición específica.
        :param formula: Fórmula temporal (puede ser un literal o una instancia de TemporalFormula).
        :param trace: Lista de estados del sistema (cada estado es un diccionario de variables y sus valores).
        :param position: Posición inicial en el rastro desde donde evaluar la fórmula.
        :return: True si la fórmula se cumple, False en caso contrario.
        """
        if isinstance(formula, str):
            # Si la fórmula es un literal (por ejemplo, 'running'), verifica su valor en el estado actual
            return trace[position].get(formula, False)
        
        # Evaluación para el operador G (siempre)
        if formula.operator == TemporalOperator.GLOBALLY:
            # Verifica que la subfórmula sea verdadera en todos los estados desde la posición actual
            return all(self.evaluate(formula.subformulas[0], trace, i) 
                       for i in range(position, len(trace)))
        
        # Evaluación para el operador F (eventualmente)
        if formula.operator == TemporalOperator.FINALLY:
            # Verifica que la subfórmula sea verdadera en al menos un estado desde la posición actual
            return any(self.evaluate(formula.subformulas[0], trace, i) 
                       for i in range(position, len(trace)))
        
        # Evaluación para el operador X (siguiente)
        if formula.operator == TemporalOperator.NEXT:
            # Verifica que la subfórmula sea verdadera en el siguiente estado
            if position + 1 >= len(trace):
                return False  # No hay un siguiente estado
            return self.evaluate(formula.subformulas[0], trace, position + 1)
        
        # Evaluación para el operador U (hasta)
        if formula.operator == TemporalOperator.UNTIL:
            # Verifica que la primera subfórmula sea verdadera hasta que la segunda lo sea
            for i in range(position, len(trace)):
                if self.evaluate(formula.subformulas[1], trace, i):
                    return True  # La segunda subfórmula se cumple
                if not self.evaluate(formula.subformulas[0], trace, i):
                    return False  # La primera subfórmula deja de cumplirse antes de que la segunda sea verdadera
            return False  # La segunda subfórmula nunca se cumple

    def parse_formula(self, expression: str) -> TemporalFormula:
        """
        Convierte una fórmula en texto a una instancia de TemporalFormula.
        :param expression: Fórmula en texto (por ejemplo, "G(running)").
        :return: Instancia de TemporalFormula que representa la fórmula.
        """
        expression = expression.strip()  # Elimina espacios en blanco alrededor de la fórmula
        
        # Parseo para el operador G (siempre)
        if expression.startswith('G(') and expression.endswith(')'):
            return TemporalFormula(
                TemporalOperator.GLOBALLY,
                self.parse_formula(expression[2:-1]))  # Recursivamente parsea la subfórmula
        
        # Parseo para el operador F (eventualmente)
        elif expression.startswith('F(') and expression.endswith(')'):
            return TemporalFormula(
                TemporalOperator.FINALLY,
                self.parse_formula(expression[2:-1]))
        
        # Parseo para el operador X (siguiente)
        elif expression.startswith('X(') and expression.endswith(')'):
            return TemporalFormula(
                TemporalOperator.NEXT,
                self.parse_formula(expression[2:-1]))
        
        # Parseo para el operador U (hasta)
        elif ' U ' in expression:
            parts = expression.split(' U ', 1)  # Divide la fórmula en las dos subfórmulas
            return TemporalFormula(
                TemporalOperator.UNTIL,
                self.parse_formula(parts[0]),
                self.parse_formula(parts[1]))
        
        # Si no es un operador, se asume que es un literal (por ejemplo, 'running' o 'error')
        else:
            return expression

# Ejemplo de verificación de lógica temporal
def verification_example():
    """
    Ejemplo que muestra cómo verificar fórmulas de lógica temporal sobre un rastro de ejecución.
    """
    verifier = TemporalVerifier()  # Crea una instancia del verificador
    
    # Rastro de ejecución del sistema (lista de estados)
    trace = [
        {'running': True, 'error': False},  # Estado 0
        {'running': True, 'error': False},  # Estado 1
        {'running': False, 'error': True},  # Estado 2
        {'running': False, 'error': False}, # Estado 3
        {'running': True, 'error': False}   # Estado 4
    ]
    
    # Fórmulas LTL a verificar
    formulas = [
        "G(running)",                # Siempre 'running' es verdadero
        "F(error)",                  # Eventualmente 'error' será verdadero
        "G(running -> X(running))",  # Si 'running' es verdadero, en el siguiente estado también lo será
        "(running U error)",         # 'running' se mantiene verdadero hasta que 'error' sea verdadero
        "G(error -> F(running))"     # Si ocurre un 'error', eventualmente 'running' será verdadero
    ]
    
    # Imprime el rastro del sistema
    print("=== Temporal Logic Verification ===")
    print("System trace:")
    for i, state in enumerate(trace):
        print(f"Step {i}: {state}")
    
    # Verifica cada fórmula y muestra los resultados
    print("\nVerification results:")
    for f in formulas:
        formula = verifier.parse_formula(f)  # Convierte la fórmula en texto a una instancia de TemporalFormula
        result = verifier.evaluate(formula, trace)  # Evalúa la fórmula en el rastro
        print(f"{str(formula):<30} -> {'✅' if result else '❌'}")

# Punto de entrada principal
if __name__ == "__main__":
    verification_example()  # Llama al ejemplo de verificación