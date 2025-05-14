def eliminar_implicaciones(formula):
    """
    Elimina las implicaciones (→) de una fórmula lógica utilizando la equivalencia lógica:
    p → q ≡ ¬p ∨ q.
    
    Args:
        formula: Una fórmula lógica representada como una lista anidada. Por ejemplo:
                 ['→', 'p', 'q'] representa "p implica q".
    
    Returns:
        La fórmula lógica sin implicaciones, transformada según la equivalencia lógica.
    """
    if isinstance(formula, str):  # Si la fórmula es una variable proposicional (cadena), devolverla tal cual.
        return formula
    if formula[0] == '→':  # Si la fórmula es una implicación.
        # Aplicar la equivalencia lógica p → q ≡ ¬p ∨ q.
        return ['∨', ['¬', eliminar_implicaciones(formula[1])], eliminar_implicaciones(formula[2])]
    # Si no es una implicación, procesar recursivamente los subcomponentes de la fórmula.
    return [formula[0]] + [eliminar_implicaciones(f) for f in formula[1:]]

def aplicar_leyes_de_morgan(formula):
    """
    Aplica las leyes de De Morgan para transformar negaciones de conjunciones o disyunciones:
    - ¬(p ∧ q) ≡ ¬p ∨ ¬q
    - ¬(p ∨ q) ≡ ¬p ∧ ¬q
    
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    
    Returns:
        La fórmula lógica con las leyes de De Morgan aplicadas.
    """
    if isinstance(formula, str):  # Si la fórmula es una variable proposicional, devolverla tal cual.
        return formula
    if formula[0] == '¬':  # Si la fórmula es una negación.
        if isinstance(formula[1], list):  # Si lo que se niega es una subfórmula (lista).
            if formula[1][0] == '∧':  # Si se niega una conjunción.
                # Aplicar ¬(p ∧ q) ≡ ¬p ∨ ¬q.
                return ['∨'] + [aplicar_leyes_de_morgan(['¬', f]) for f in formula[1][1:]]
            elif formula[1][0] == '∨':  # Si se niega una disyunción.
                # Aplicar ¬(p ∨ q) ≡ ¬p ∧ ¬q.
                return ['∧'] + [aplicar_leyes_de_morgan(['¬', f]) for f in formula[1][1:]]
    # Si no es una negación o no aplica De Morgan, procesar recursivamente los subcomponentes.
    return [formula[0]] + [aplicar_leyes_de_morgan(f) for f in formula[1:]]

def distribuir_disjunciones(formula):
    """
    Distribuye disyunciones (∨) sobre conjunciones (∧) para convertir la fórmula a Forma Normal Conjuntiva (FNC):
    - p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r).
    
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    
    Returns:
        La fórmula lógica con las disyunciones distribuidas.
    """
    if isinstance(formula, str):  # Si la fórmula es una variable proposicional, devolverla tal cual.
        return formula
    if formula[0] == '∨':  # Si la fórmula es una disyunción.
        # Buscar si algún operando de la disyunción es una conjunción.
        for i, f in enumerate(formula[1:]):
            if isinstance(f, list) and f[0] == '∧':  # Si se encuentra una conjunción.
                # Aplicar la distribución: p ∨ (q ∧ r) ≡ (p ∨ q) ∧ (p ∨ r).
                otro = formula[1:i+1] + formula[i+2:]  # Otros elementos de la disyunción.
                return ['∧', distribuir_disjunciones(['∨', formula[1], f[1]]), 
                               distribuir_disjunciones(['∨', formula[1], f[2]])]
    # Si no hay conjunciones en la disyunción, procesar recursivamente los subcomponentes.
    return [formula[0]] + [distribuir_disjunciones(f) for f in formula[1:]]

def a_fnc(formula):
    """
    Convierte una fórmula lógica a Forma Normal Conjuntiva (FNC).
    La FNC es una conjunción de disyunciones, donde cada disyunción contiene literales.
    
    Args:
        formula: Una fórmula lógica representada como una lista anidada.
    
    Returns:
        La fórmula lógica transformada a FNC.
    """
    # Aplicar las transformaciones en el orden adecuado:
    # 1. Eliminar implicaciones.
    # 2. Aplicar las leyes de De Morgan.
    # 3. Distribuir disyunciones sobre conjunciones.
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
    resolvente = None  # Inicializar el resolvente como None.
    for literal in clausula1:  # Iterar sobre los literales de la primera cláusula.
        # Buscar literales complementarios (p y ¬p).
        if f"¬{literal}" in clausula2:  # Caso 1: literal está en clausula1 y su negación en clausula2.
            # Crear el resolvente eliminando el literal y su complemento.
            resolvente = [l for l in clausula1 if l != literal] + [l for l in clausula2 if l != f"¬{literal}"]
            break
        elif literal.startswith("¬") and literal[1:] in clausula2:  # Caso 2: ¬p está en clausula1 y p en clausula2.
            # Crear el resolvente eliminando el literal y su complemento.
            resolvente = [l for l in clausula1 if l != literal] + [l for l in clausula2 if l != literal[1:]]
            break
    return resolvente  # Devolver la nueva cláusula resolvente o None si no se pudo resolver.

# Ejemplo de uso
if __name__ == "__main__":
    # Fórmula inicial: (p → q) ∧ (q → r).
    formula = ['∧', ['→', 'p', 'q'], ['→', 'q', 'r']]
    
    # Convertir la fórmula a Forma Normal Conjuntiva (FNC).
    fnc = a_fnc(formula)
    print("FNC:", fnc)  # Salida esperada: ['∧', ['∨', '¬p', 'q'], ['∨', '¬q', 'r']]

    # Resolución: ¿Se puede inferir p → r?
    clausulas = [['¬p', 'q'], ['¬q', 'r'], ['p'], ['¬r']]  # Incluimos ¬r para probar contradicción.
    nueva_clausula = resolver(clausulas[0], clausulas[1])
    print("Resolvente:", nueva_clausula)  # Salida esperada: ['¬p', 'r']