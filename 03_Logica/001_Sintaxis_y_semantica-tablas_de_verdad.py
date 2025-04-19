def tabla_de_verdad(variables, expresion):
    """
    Genera una tabla de verdad para una expresión lógica dada.
    
    Args:
        variables (list): Lista de variables proposicionales (e.g., ['p', 'q']).
        expresion (function): Función que evalúa la expresión lógica (e.g., lambda p, q: p and q).
    """
    n = len(variables)  # Número de variables proposicionales
    combinaciones = []  # Lista para almacenar todas las combinaciones de valores de verdad

    # Generar todas las combinaciones posibles de valores de verdad (2^n combinaciones)
    for i in range(2**n):
        valores = []  # Lista para almacenar los valores de verdad de una combinación
        for j in range(n):
            # Determinar el valor de verdad (True o False) según el bit correspondiente
            # Se utiliza un desplazamiento de bits para obtener el valor de cada variable
            valores.append(bool((i >> (n - 1 - j)) & 1))
        combinaciones.append(valores)  # Agregar la combinación generada a la lista

    # Imprimir el encabezado de la tabla
    header = " | ".join(variables) + " | Resultado"  # Encabezado con nombres de variables y columna de resultado
    print(header)
    print("-" * len(header))  # Línea separadora para la tabla

    # Evaluar la expresión lógica para cada combinación de valores de verdad
    for valores in combinaciones:
        # Evaluar la expresión lógica con los valores actuales
        resultado = expresion(*valores)
        # Crear una fila de la tabla con los valores de verdad y el resultado
        fila = " | ".join(["V" if v else "F" for v in valores]) + f" |   {'V' if resultado else 'F'}"
        print(fila)  # Imprimir la fila de la tabla

# Ejemplo de uso:
if __name__ == "__main__":
    # Definir las variables proposicionales y la expresión lógica
    # En este caso, la expresión lógica es (p ∧ q) → r, que se representa como: not (p and q) or r
    variables = ['p', 'q', 'r']  # Lista de variables proposicionales
    expresion = lambda p, q, r: not (p and q) or r  # Expresión lógica como función lambda
    
    # Generar e imprimir la tabla de verdad para la expresión lógica
    tabla_de_verdad(variables, expresion)