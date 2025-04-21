# --- Función para verificar si algo es una variable (por ejemplo, 'x', 'y')
# Una variable se define como una cadena en minúsculas.
def es_variable(x):
    return isinstance(x, str) and x.islower()

# --- Sustituye variables con su valor si está en el diccionario de sustitución
# Esta función toma una expresión y un diccionario de sustituciones.
# Si la expresión es una variable, intenta reemplazarla con su valor en el diccionario.
# Si es una tupla, aplica la sustitución recursivamente a cada elemento.
def sustituir(expr, sustituciones):
    if es_variable(expr):  # Si es una variable, busca su valor en el diccionario.
        return sustituciones.get(expr, expr)
    elif isinstance(expr, tuple):  # Si es una tupla, aplica sustitución a cada elemento.
        return tuple(sustituir(e, sustituciones) for e in expr)
    else:  # Si no es una variable ni una tupla, devuelve la expresión tal cual.
        return expr

# --- Intenta unificar dos expresiones lógicas
# La unificación busca encontrar un conjunto de sustituciones que hagan iguales dos expresiones.
# Si no es posible unificar, devuelve None.
def unificar(e1, e2, sustituciones=None):
    if sustituciones is None:  # Inicializa el diccionario de sustituciones si no se proporciona.
        sustituciones = {}

    # Aplica las sustituciones actuales a ambas expresiones.
    e1 = sustituir(e1, sustituciones)
    e2 = sustituir(e2, sustituciones)

    # Caso base: si las expresiones ya son iguales, no se necesitan más sustituciones.
    if e1 == e2:
        return sustituciones

    # Si la primera expresión es una variable, se agrega una sustitución.
    if es_variable(e1):
        sustituciones[e1] = e2
        return sustituciones

    # Si la segunda expresión es una variable, se agrega una sustitución.
    if es_variable(e2):
        sustituciones[e2] = e1
        return sustituciones

    # Si ambas expresiones son tuplas del mismo tamaño, intenta unificar elemento por elemento.
    if isinstance(e1, tuple) and isinstance(e2, tuple) and len(e1) == len(e2):
        for a, b in zip(e1, e2):  # Itera sobre los elementos de ambas tuplas.
            sustituciones = unificar(a, b, sustituciones)
            if sustituciones is None:  # Si alguna unificación falla, devuelve None.
                return None
        return sustituciones

    # Si no se cumple ninguno de los casos anteriores, no se pueden unificar.
    return None

# --- Ejemplo de uso
if __name__ == "__main__":
    # Definimos dos expresiones lógicas para unificar.
    # En este caso, expresion1 contiene una variable 'X' que debe ser unificada con 'maria' en expresion2.
    expresion1 = ("padre", "juan", "X")
    expresion2 = ("padre", "juan", "maria")

    # Llamamos a la función de unificación.
    resultado = unificar(expresion1, expresion2)

    # Mostramos los resultados de la unificación.
    print("Unificación de expresiones:")
    print(f"  {expresion1}")
    print(f"  {expresion2}")
    if resultado:
        print("\n Sustituciones encontradas:")
        # Iteramos sobre el diccionario de sustituciones y las imprimimos.
        for var, val in resultado.items():
            print(f"  {var} = {val}")
    else:
        print("\n No se pudo unificar.")
