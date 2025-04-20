def eliminar_implicaciones(formula):
    """
    Elimina las implicaciones (→) de una fórmula lógica utilizando la equivalencia:
    p → q ≡ ¬p ∨ q.
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    Returns:
        La fórmula sin implicaciones.
    """
    if isinstance(formula, str):  # Si es una variable proposicional, devolverla tal cual.
        return formula
    if formula[0] == '→':  # Si es una implicación, aplicar la equivalencia.
        return ['∨', ['¬', eliminar_implicaciones(formula[1])], eliminar_implicaciones(formula[2])]
    # Procesar recursivamente los subcomponentes de la fórmula.
    return [formula[0]] + [eliminar_implicaciones(f) for f in formula[1:]]

def aplicar_leyes_de_morgan(formula):
    """
    Aplica las leyes de De Morgan para transformar negaciones de conjunciones o disyunciones:
    ¬(p ∧ q) ≡ ¬p ∨ ¬q
    ¬(p ∨ q) ≡ ¬p ∧ ¬q
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    Returns:
        La fórmula con las leyes de De Morgan aplicadas.
    """
    if isinstance(formula, str):  # Si es una variable proposicional, devolverla tal cual.
        return formula
    if formula[0] == '¬':  # Si es una negación.
        if isinstance(formula[1], list):  # Si lo que se niega es una lista (subfórmula).
            if formula[1][0] == '∧':  # Negación de una conjunción.
                return ['∨'] + [aplicar_leyes_de_morgan(['¬', f]) for f in formula[1][1:]]
            elif formula[1][0] == '∨':  # Negación de una disyunción.
                return ['∧'] + [aplicar_leyes_de_morgan(['¬', f]) for f in formula[1][1:]]
    # Procesar recursivamente los subcomponentes de la fórmula.
    return [formula[0]] + [aplicar_leyes_de_morgan(f) for f in formula[1:]]

def distribuir_disjunciones(formula):
    """
    Distribuye disyunciones (∨) sobre conjunciones (∧) para convertir la fórmula a FNC:
    p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r).
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    Returns:
        La fórmula con las disyunciones distribuidas.
    """
    if isinstance(formula, str):  # Si es una variable proposicional, devolverla tal cual.
        return formula
    if formula[0] == '∨':  # Si es una disyunción.
        # Buscar si algún operando es una conjunción (∧).
        for i, f in enumerate(formula[1:]):
            if isinstance(f, list) and f[0] == '∧':
                # Distribuir: f ∨ (g ∧ h) → (f ∨ g) ∧ (f ∨ h).
                otro = formula[1:i+1] + formula[i+2:]
                return ['∧', distribuir_disjunciones(['∨', formula[1], f[1]]), 
                               distribuir_disjunciones(['∨', formula[1], f[2]])]
    # Procesar recursivamente los subcomponentes de la fórmula.
    return [formula[0]] + [distribuir_disjunciones(f) for f in formula[1:]]

def a_fnc(formula):
    """
    Convierte una fórmula lógica a Forma Normal Conjuntiva (FNC).
    La FNC es una conjunción de disyunciones.
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    Returns:
        La fórmula en FNC.
    """
    # Aplicar las transformaciones en el orden adecuado.
    return distribuir_disjunciones(aplicar_leyes_de_morgan(eliminar_implicaciones(formula)))

def resolver(clausula1, clausula2):
    """
    Aplica la regla de resolución a dos cláusulas para derivar una nueva cláusula.
    La regla de resolución elimina un par de literales complementarios (p y ¬p).
    Args:
        clausula1: Lista de literales (ej. ['p', '¬q']).
        clausula2: Lista de literales (ej. ['¬p', 'r']).
    Returns:
        Una nueva cláusula resultante de la resolución, o None si no se puede resolver.
    """
    resolvente = None
    for literal in clausula1:
        # Buscar literales complementarios (p y ¬p).
        if f"¬{literal}" in clausula2:
            resolvente = [l for l in clausula1 if l != literal] + [l for l in clausula2 if l != f"¬{literal}"]
            break
        elif literal.startswith("¬") and literal[1:] in clausula2:
            resolvente = [l for l in clausula1 if l != literal] + [l for l in clausula2 if l != literal[1:]]
            break
    return resolvente

# Ejemplo de uso
if __name__ == "__main__":
    # Fórmula: (p → q) ∧ (q → r).
    formula = ['∧', ['→', 'p', 'q'], ['→', 'q', 'r']]
    # Convertir la fórmula a Forma Normal Conjuntiva (FNC).
    fnc = a_fnc(formula)
    print("FNC:", fnc)  # Salida esperada: ['∧', ['∨', '¬p', 'q'], ['∨', '¬q', 'r']]

    # Resolución: ¿Se puede inferir p → r?
    clausulas = [['¬p', 'q'], ['¬q', 'r'], ['p'], ['¬r']]  # Incluimos ¬r para probar contradicción.
    nueva_clausula = resolver(clausulas[0], clausulas[1])
    print("Resolvente:", nueva_clausula)  # Salida esperada: ['¬p', 'r']