from enum import Enum, auto
from typing import List, Dict, Union

# Enumeración para representar los operadores temporales de LTL
class TemporalOperator(Enum):
    GLOBALLY = auto()  # G (siempre)
    FINALLY = auto()   # F (eventualmente)
    NEXT = auto()      # X (siguiente)
    UNTIL = auto()     # U (hasta)

# Clase que representa una fórmula temporal
class TemporalFormula:
    def __init__(self, operator: TemporalOperator, *subformulas):
        self.operator = operator  # Operador temporal (G, F, X, U)
        self.subformulas = subformulas  # Subfórmulas asociadas al operador
    
    def __str__(self):
        # Mapeo de operadores a sus representaciones en texto
        op_map = {
            TemporalOperator.GLOBALLY: 'G',
            TemporalOperator.FINALLY: 'F',
            TemporalOperator.NEXT: 'X',
            TemporalOperator.UNTIL: 'U'
        }
        # Formateo de la fórmula como cadena
        if self.operator == TemporalOperator.UNTIL:
            return f"({self.subformulas[0]} U {self.subformulas[1]})"
        return f"{op_map[self.operator]}({self.subformulas[0]})"

# Clase que verifica fórmulas temporales sobre un rastro de ejecución
class TemporalVerifier:
    def evaluate(self, formula: Union[str, TemporalFormula], trace: List[Dict[str, bool]], position: int = 0) -> bool:
        """
        Evalúa una fórmula temporal en un rastro de ejecución desde una posición específica.
        """
        if isinstance(formula, str):
            # Si la fórmula es un literal, verifica su valor en el estado actual
            return trace[position].get(formula, False)
        
        # Evaluación para el operador G (siempre)
        if formula.operator == TemporalOperator.GLOBALLY:
            return all(self.evaluate(formula.subformulas[0], trace, i) 
                   for i in range(position, len(trace)))
        
        # Evaluación para el operador F (eventualmente)
        if formula.operator == TemporalOperator.FINALLY:
            return any(self.evaluate(formula.subformulas[0], trace, i) 
                   for i in range(position, len(trace)))
        
        # Evaluación para el operador X (siguiente)
        if formula.operator == TemporalOperator.NEXT:
            if position + 1 >= len(trace):
                return False  # No hay un siguiente estado
            return self.evaluate(formula.subformulas[0], trace, position + 1)
        
        # Evaluación para el operador U (hasta)
        if formula.operator == TemporalOperator.UNTIL:
            for i in range(position, len(trace)):
                if self.evaluate(formula.subformulas[1], trace, i):
                    return True  # Se cumple la segunda subfórmula
                if not self.evaluate(formula.subformulas[0], trace, i):
                    return False  # La primera subfórmula deja de cumplirse
            return False

    def parse_formula(self, expression: str) -> TemporalFormula:
        """
        Convierte una fórmula en texto a una instancia de TemporalFormula.
        """
        expression = expression.strip()
        
        # Parseo para el operador G (siempre)
        if expression.startswith('G(') and expression.endswith(')'):
            return TemporalFormula(
                TemporalOperator.GLOBALLY,
                self.parse_formula(expression[2:-1]))
        
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
            parts = expression.split(' U ', 1)
            return TemporalFormula(
                TemporalOperator.UNTIL,
                self.parse_formula(parts[0]),
                self.parse_formula(parts[1]))
        
        # Si no es un operador, se asume que es un literal
        else:
            return expression

# Ejemplo de verificación de lógica temporal
def verification_example():
    verifier = TemporalVerifier()
    
    # Rastro de ejecución del sistema (lista de estados)
    trace = [
        {'running': True, 'error': False},
        {'running': True, 'error': False},
        {'running': False, 'error': True},
        {'running': False, 'error': False},
        {'running': True, 'error': False}
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
    verification_example()