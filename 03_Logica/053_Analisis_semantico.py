def semantic_analysis_manual(sentence):
    """
    Realiza un análisis semántico manual de una oración en español.
    Identifica el sujeto, el verbo y el objeto, y los representa en una fórmula lógica.
    
    Parámetros:
        sentence (str): La oración a analizar.
    
    Retorna:
        str: Una fórmula lógica que representa la estructura de la oración,
             o un mensaje indicando que no se pudo analizar.
    """
    
    # Diccionario de verbos comunes con sus lemas (para lematización manual)
    verb_lemmas = {
        'persigue': 'perseguir',  # Ejemplo: "persigue" se lematiza a "perseguir"
        'come': 'comer',          # Ejemplo: "come" se lematiza a "comer"
        'observa': 'observar'     # Ejemplo: "observa" se lematiza a "observar"
    }
    
    # Convertir la oración a minúsculas y dividirla en palabras
    words = sentence.lower().split()
    
    # Inicializar las variables para el sujeto, el verbo y el objeto
    subject = None  # Sujeto de la oración (primer sustantivo encontrado)
    verb = None     # Verbo de la oración (encontrado en el diccionario de lemas)
    obj = None      # Objeto de la oración (último sustantivo encontrado)

    # Buscar el sujeto (primer sustantivo en la lista de palabras)
    for word in words:
        if word in ['gato', 'perro', 'ratón', 'sol']:  # Lista de sustantivos conocidos
            subject = word
            break  # Detener la búsqueda al encontrar el primer sustantivo
    
    # Buscar el verbo (palabra que coincide con una clave en el diccionario de lemas)
    for word in words:
        if word in verb_lemmas:
            verb = verb_lemmas[word]  # Obtener el lema del verbo
            break  # Detener la búsqueda al encontrar el primer verbo
    
    # Buscar el objeto (último sustantivo en la lista de palabras, distinto del sujeto)
    for word in reversed(words):  # Iterar desde el final de la lista
        if word in ['gato', 'perro', 'ratón', 'sol'] and word != subject:
            obj = word
            break  # Detener la búsqueda al encontrar el primer sustantivo válido desde el final
    
    # Si se encontraron sujeto, verbo y objeto, construir la fórmula lógica
    if subject and verb and obj:
        return f"Fórmula lógica: {subject}(x) & {verb}(x, y) & {obj}(y)"
    
    # Si no se pudo analizar la estructura, devolver un mensaje de error
    return "No se pudo analizar la estructura"

# Ejemplo de uso
sentence = "el gato persigue al ratón"  # Oración de ejemplo
print("Oración:", sentence)  # Imprimir la oración original
print("Resultado:", semantic_analysis_manual(sentence))  # Imprimir el resultado del análisis