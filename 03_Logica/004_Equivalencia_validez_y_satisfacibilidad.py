def es_tautologia(formula, variables):
    """Verifica si una fórmula es válida (tautología).
    Una fórmula es una tautología si es verdadera para todas las combinaciones posibles de valores de verdad."""
    for valores in generar_combinaciones(variables):  # Generar todas las combinaciones de valores de verdad
        if not evaluar(formula, dict(zip(variables, valores))):  # Evaluar la fórmula con los valores actuales
            return False  # Si hay al menos un caso falso, no es una tautología
    return True  # Si es verdadera en todos los casos, es una tautología

def es_satisfacible(formula, variables):
    """Verifica si una fórmula es satisfacible.
    Una fórmula es satisfacible si es verdadera para al menos una combinación de valores de verdad."""
    for valores in generar_combinaciones(variables):  # Generar todas las combinaciones de valores de verdad
        if evaluar(formula, dict(zip(variables, valores))):  # Evaluar la fórmula con los valores actuales
            return True  # Si hay al menos un caso verdadero, es satisfacible
    return False  # Si no hay ningún caso verdadero, no es satisfacible

def son_equivalentes(formula1, formula2, variables):
    """Verifica si dos fórmulas son lógicamente equivalentes.
    Dos fórmulas son equivalentes si tienen el mismo valor de verdad para todas las combinaciones posibles."""
    for valores in generar_combinaciones(variables):  # Generar todas las combinaciones de valores de verdad
        # Evaluar ambas fórmulas con los mismos valores y comparar los resultados
        if evaluar(formula1, dict(zip(variables, valores))) != evaluar(formula2, dict(zip(variables, valores))):
            return False  # Si hay al menos un caso donde difieren, no son equivalentes
    return True  # Si son iguales en todos los casos, son equivalentes

# Funciones auxiliares
def generar_combinaciones(variables):
    """Genera todas las combinaciones de valores de verdad para las variables.
    Por ejemplo, para ['p', 'q'], genera: [False, False], [False, True], [True, False], [True, True]."""
    n = len(variables)  # Número de variables
    for i in range(2**n):  # Iterar sobre todas las combinaciones posibles (2^n)
        # Generar una combinación de valores de verdad como una lista de booleanos
        yield [bool((i >> j) & 1) for j in reversed(range(n))]

def evaluar(formula, contexto):
    """Evalúa una fórmula lógica dado un contexto (diccionario de variables).
    Por ejemplo, si formula = lambda p, q: (p and not q) or q y contexto = {'p': True, 'q': False},
    evalúa la fórmula con esos valores."""
    return formula(**contexto)  # Desempaquetar el contexto como argumentos de la fórmula

# Ejemplo de uso
if __name__ == "__main__":
    # Definir variables y fórmulas
    variables = ['p', 'q']  # Lista de variables proposicionales
    formula1 = lambda p, q: (p or q)  # Fórmula lógica: p ∨ q
    formula2 = lambda p, q: not (not p and not q)  # Fórmula lógica equivalente: ¬(¬p ∧ ¬q)
    formula3 = lambda p, q: p and not p  # Fórmula lógica insatisfacible: p ∧ ¬p

    # Verificar equivalencia entre formula1 y formula2
    print("¿Formula1 ≡ Formula2?", son_equivalentes(formula1, formula2, variables))  # True, son equivalentes

    # Verificar validez (tautología) de formula1
    print("¿Formula1 es tautología?", es_tautologia(formula1, variables))  # False, no es una tautología

    # Verificar satisfacibilidad de formula3
    print("¿Formula3 es satisfacible?", es_satisfacible(formula3, variables))  # False, no es satisfacible