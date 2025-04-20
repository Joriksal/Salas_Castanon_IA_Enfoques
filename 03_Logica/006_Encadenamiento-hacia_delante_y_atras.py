def backward_chaining(hechos, reglas, objetivo, ruta=None):
    """
    Implementa el algoritmo de encadenamiento hacia atrás para verificar si un objetivo
    puede ser inferido a partir de un conjunto de hechos y reglas.

    Args:
        hechos (set): Conjunto de hechos conocidos (ej: {"p", "q"}).
        reglas (list): Lista de reglas en forma de tuplas (premisas, conclusión).
                      Ejemplo: [({"p", "q"}, "r"), ({"r"}, "s")].
        objetivo (str): Hecho que se desea verificar (ej: "t").
        ruta (list): Lista opcional para rastrear la cadena de inferencias (útil para debug).

    Returns:
        bool: True si el objetivo puede ser inferido, False en caso contrario.
    """
    # Inicializa la ruta si no se proporciona (para rastrear la inferencia).
    if ruta is None:
        ruta = []

    # Si el objetivo ya está en los hechos conocidos, se puede inferir directamente.
    if objetivo in hechos:
        return True

    # Recorre todas las reglas para buscar una cuya conclusión sea el objetivo.
    for premisas, conclusion in reglas:
        if conclusion == objetivo:
            # Imprime la regla utilizada para debug.
            print(f"Regla usada: {premisas} → {conclusion}")

            # Verifica si todas las premisas de la regla son verdaderas.
            todos_verdaderos = True
            for premisa in premisas:
                # Llama recursivamente para verificar cada premisa.
                if not backward_chaining(hechos, reglas, premisa, ruta + [objetivo]):
                    todos_verdaderos = False
                    break  # Si una premisa no es verdadera, detiene la verificación.

            # Si todas las premisas son verdaderas, el objetivo puede ser inferido.
            if todos_verdaderos:
                return True

    # Si no se encuentra ninguna regla que permita inferir el objetivo, retorna False.
    return False

# Ejemplo de uso del algoritmo
if __name__ == "__main__":
    # Definición de las reglas en forma de tuplas (premisas, conclusión).
    reglas = [
        ({"p", "q"}, "r"),  # Si "p" y "q" son verdaderos, entonces "r" es verdadero.
        ({"r"}, "s"),       # Si "r" es verdadero, entonces "s" es verdadero.
        ({"s"}, "t")        # Si "s" es verdadero, entonces "t" es verdadero.
    ]

    # Conjunto de hechos conocidos.
    hechos = {"p", "q"}  # Sabemos que "p" y "q" son verdaderos.

    # Objetivo que queremos verificar.
    objetivo = "t"  # Queremos saber si "t" puede ser inferido.

    # Llamada al algoritmo de encadenamiento hacia atrás.
    print(f"¿Se puede inferir '{objetivo}'? (Backward Chaining):", 
          backward_chaining(hechos, reglas, objetivo))
    # Salida esperada: True, ya que "t" puede ser inferido a partir de los hechos y reglas.