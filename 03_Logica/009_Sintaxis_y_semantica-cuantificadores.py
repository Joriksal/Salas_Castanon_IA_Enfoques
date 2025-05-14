def evaluar_formula(formula, dominio, interpretacion):
    """
    Evalúa una fórmula lógica con cuantificadores (∀, ∃) en un dominio finito.

    Args:
        formula: Diccionario que representa la fórmula lógica. Ejemplo:
                 {"tipo": "∀", "var": "x", "subformula": ...}.
        dominio: Lista de elementos del dominio sobre los que se evalúa la fórmula.
                 Ejemplo: [1, 2, 3].
        interpretacion: Función que evalúa predicados atómicos. Ejemplo:
                        lambda atomica: atomica == ("P", 3).

    Returns:
        True o False dependiendo de si la fórmula es verdadera en el dominio.
    """
    # Extraemos el tipo de la fórmula (puede ser atómica, cuantificador o conector lógico).
    tipo = formula["tipo"]

    # Caso base: si la fórmula es atómica (ejemplo: P(x)).
    if tipo == "atomica":
        # Evaluamos el predicado atómico utilizando la función de interpretación.
        return interpretacion(formula["predicado"])

    # Caso: conectores lógicos.
    elif tipo == "no":  # Negación lógica (¬φ).
        # Evaluamos la subfórmula y devolvemos su negación.
        return not evaluar_formula(formula["subformula"], dominio, interpretacion)
    elif tipo == "y":  # Conjunción lógica (φ ∧ ψ).
        # Evaluamos ambas subfórmulas y devolvemos True si ambas son verdaderas.
        return evaluar_formula(formula["subformula1"], dominio, interpretacion) and \
               evaluar_formula(formula["subformula2"], dominio, interpretacion)
    elif tipo == "o":  # Disyunción lógica (φ ∨ ψ).
        # Evaluamos ambas subfórmulas y devolvemos True si al menos una es verdadera.
        return evaluar_formula(formula["subformula1"], dominio, interpretacion) or \
               evaluar_formula(formula["subformula2"], dominio, interpretacion)

    # Caso: cuantificador universal (∀x φ).
    elif tipo == "∀":
        # Iteramos sobre todos los elementos del dominio.
        for elemento in dominio:
            # Sustituimos la variable en la subfórmula por el elemento actual del dominio.
            subformula_evaluada = sustituir_variable(formula["subformula"], formula["var"], elemento)
            # Si la subfórmula no es verdadera para algún elemento, retornamos False.
            if not evaluar_formula(subformula_evaluada, dominio, interpretacion):
                return False
        # Si es verdadera para todos los elementos, retornamos True.
        return True

    # Caso: cuantificador existencial (∃x φ).
    elif tipo == "∃":
        # Iteramos sobre todos los elementos del dominio.
        for elemento in dominio:
            # Sustituimos la variable en la subfórmula por el elemento actual del dominio.
            subformula_evaluada = sustituir_variable(formula["subformula"], formula["var"], elemento)
            # Si la subfórmula es verdadera para algún elemento, retornamos True.
            if evaluar_formula(subformula_evaluada, dominio, interpretacion):
                return True
        # Si no es verdadera para ningún elemento, retornamos False.
        return False


def sustituir_variable(formula, variable, valor):
    """
    Sustituye una variable en una fórmula por un valor concreto.

    Args:
        formula: Diccionario que representa la fórmula lógica.
        variable: Nombre de la variable a sustituir (ejemplo: "x").
        valor: Valor con el que se sustituirá la variable (ejemplo: 3).

    Returns:
        Una nueva fórmula con la variable sustituida por el valor.
    """
    # Caso base: si la fórmula es atómica.
    if formula["tipo"] == "atomica":
        # Reemplazamos la variable en el predicado atómico.
        # Ejemplo: ("P", "x") → ("P", 3).
        nuevo_predicado = tuple(valor if x == variable else x for x in formula["predicado"])
        return {"tipo": "atomica", "predicado": nuevo_predicado}
    else:
        # Caso recursivo: aplicamos la sustitución a las subfórmulas.
        nueva_formula = formula.copy()
        if "subformula" in nueva_formula:
            nueva_formula["subformula"] = sustituir_variable(formula["subformula"], variable, valor)
        if "subformula1" in nueva_formula:
            nueva_formula["subformula1"] = sustituir_variable(formula["subformula1"], variable, valor)
        if "subformula2" in nueva_formula:
            nueva_formula["subformula2"] = sustituir_variable(formula["subformula2"], variable, valor)
        return nueva_formula


# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Dominio: {1, 2, 3}.
    # Este es el conjunto de valores sobre los que se evaluará la fórmula.
    dominio = [1, 2, 3]
    
    # Fórmula: ∃x (x > 2).
    # Representa que existe al menos un x en el dominio tal que x > 2.
    formula = {
        "tipo": "∃",  # Cuantificador existencial.
        "var": "x",  # Variable cuantificada.
        "subformula": {  # Subfórmula que se evalúa para cada elemento del dominio.
            "tipo": "atomica",
            "predicado": (">", "x", 2)  # Representa "x > 2".
        }
    }
    
    # Interpretación: Define cómo se evalúan los predicados atómicos.
    # En este caso, ">" se interpreta como la operación mayor que.
    def interpretacion(predicado):
        # Verificamos si el predicado es una comparación "mayor que".
        if predicado[0] == ">":
            # Evaluamos si el primer elemento es mayor que el segundo.
            return predicado[1] > predicado[2]  # Ejemplo: (">", 3, 2) → True.
    
    # Evaluación de la fórmula en el dominio con la interpretación dada.
    resultado = evaluar_formula(formula, dominio, interpretacion)
    # Mostramos el resultado de la evaluación.
    print(f"La fórmula ∃x (x > 2) es {resultado} en el dominio {dominio}")
    # Salida esperada: "La fórmula ∃x (x > 2) es True en el dominio [1, 2, 3]"