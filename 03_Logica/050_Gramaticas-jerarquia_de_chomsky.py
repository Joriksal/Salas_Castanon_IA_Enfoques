def classify_grammar(rules):
    """
    Clasifica una gramática según la jerarquía de Chomsky:
    - Tipo 3: Gramática Regular
    - Tipo 2: Gramática Libre de Contexto
    - Tipo 1: Gramática Sensible al Contexto
    - Tipo 0: Gramática Irrestrica

    Parámetros:
    - rules: Lista de reglas de producción en forma de cadenas "izquierda -> derecha".

    Retorna:
    - Una cadena indicando el tipo de gramática.
    """
    # Inicializamos los indicadores para cada tipo de gramática
    is_type3 = True  # Gramática regular
    is_type2 = True  # Gramática libre de contexto
    is_type1 = True  # Gramática sensible al contexto

    # Iteramos sobre cada regla de producción
    for rule in rules:
        # Dividimos la regla en la parte izquierda y derecha del símbolo "->"
        left, right = rule.split("->")
        left = left.strip()  # Eliminamos espacios en blanco
        right = right.strip()

        # Verificamos si cumple con las condiciones de una gramática regular (Tipo 3)
        # - La parte izquierda debe ser un único símbolo no terminal (mayúscula)
        if len(left) != 1 or not left.isupper():
            is_type3 = False
        # - La parte derecha debe ser una cadena de terminales o un terminal seguido de un no terminal
        if not (right.islower() or (len(right) == 2 and right[0].islower() and right[1].isupper())):
            is_type3 = False

        # Verificamos si cumple con las condiciones de una gramática libre de contexto (Tipo 2)
        # - La parte izquierda debe ser un único símbolo no terminal
        if len(left) != 1 or not left.isupper():
            is_type2 = False

        # Verificamos si cumple con las condiciones de una gramática sensible al contexto (Tipo 1)
        # - La longitud de la parte derecha debe ser mayor o igual a la de la parte izquierda
        if len(right) < len(left):
            is_type1 = False

    # Determinamos el tipo de gramática según las verificaciones
    if is_type3:
        return "Tipo 3: Gramática Regular"
    elif is_type2:
        return "Tipo 2: Gramática Libre de Contexto"
    elif is_type1:
        return "Tipo 1: Gramática Sensible al Contexto"
    else:
        return "Tipo 0: Gramática Irrestrica"

# --- Ejemplos de prueba ---
if __name__ == "__main__":
    # Ejemplo de gramática regular (Tipo 3)
    print("Ejemplo 1 (Tipo 3):")
    grammar1 = [
        "S -> aA",  # Regla: S produce un terminal 'a' seguido de un no terminal 'A'
        "A -> b"    # Regla: A produce un terminal 'b'
    ]
    print(classify_grammar(grammar1))

    # Ejemplo de gramática libre de contexto (Tipo 2)
    print("\nEjemplo 2 (Tipo 2):")
    grammar2 = [
        "S -> aSb",  # Regla: S produce un terminal 'a', seguido de S, seguido de un terminal 'b'
        "S -> ε"     # Regla: S produce la cadena vacía
    ]
    print(classify_grammar(grammar2))

    # Ejemplo de gramática sensible al contexto (Tipo 1)
    print("\nEjemplo 3 (Tipo 1):")
    grammar3 = [
        "aS -> abS",  # Regla: 'aS' produce 'abS'
        "Sb -> bb"    # Regla: 'Sb' produce 'bb'
    ]
    print(classify_grammar(grammar3))

    # Ejemplo de gramática irrestricta (Tipo 0)
    print("\nEjemplo 4 (Tipo 0):")
    grammar4 = [
        "S -> a",      # Regla: S produce un terminal 'a'
        "abA -> B"     # Regla: 'abA' produce un no terminal 'B'
    ]
    print(classify_grammar(grammar4))
