import re  # Para trabajar con expresiones regulares, utilizadas para definir y buscar patrones de tokens en el código fuente.

# Definimos los patrones de los tokens
# Cada token tiene un nombre (como 'NUMBER', 'ASSIGN', etc.) y un patrón regex asociado
TOKEN_SPEC = [
    ('NUMBER',   r'\d+(\.\d+)?'),       # Números enteros o flotantes (ej. 42, 3.14)
    ('ASSIGN',   r'='),                 # Signo de asignación (ej. =)
    ('END',      r';'),                 # Fin de instrucción (ej. ;)
    ('ID',       r'[A-Za-z_]\w*'),      # Identificadores (ej. variables como x, y1)
    ('OP',       r'[+\-*/]'),           # Operadores aritméticos (ej. +, -, *, /)
    ('LPAREN',   r'\('),                # Paréntesis izquierdo (ej. ()
    ('RPAREN',   r'\)'),                # Paréntesis derecho (ej. ))
    ('SKIP',     r'[ \t]+'),            # Espacios y tabulaciones (se ignoran)
    ('MISMATCH', r'.'),                 # Cualquier otro carácter no reconocido
]

# Compilamos los patrones en una única expresión regular
# Esto permite buscar todos los tokens definidos en TOKEN_SPEC en una sola pasada
tok_regex = '|'.join(f'(?P<{name}>{pattern})' for name, pattern in TOKEN_SPEC)
token_compiled = re.compile(tok_regex)

def lexical_analyzer(code):
    """
    Analiza una cadena de entrada y devuelve los tokens reconocidos.
    
    Args:
        code (str): Código fuente a analizar.
    
    Returns:
        list: Lista de tuplas (tipo_de_token, valor).
    """
    tokens = []  # Lista para almacenar los tokens reconocidos
    for match in token_compiled.finditer(code):  # Iteramos sobre las coincidencias en el código
        kind = match.lastgroup  # Nombre del token (como 'NUMBER', 'ID', etc.)
        value = match.group()   # Valor del token encontrado en el código
        if kind == 'SKIP':      # Ignoramos espacios y tabulaciones
            continue
        elif kind == 'MISMATCH':  # Si encontramos un carácter no reconocido, lanzamos un error
            raise RuntimeError(f'Caracter inesperado: {value!r}')
        else:
            tokens.append((kind, value))  # Agregamos el token reconocido a la lista
    return tokens

# --- Ejemplo práctico ---
if __name__ == "__main__":
    # Código de entrada para analizar
    input_code = "x = 42 + y1;"
    print("Código de entrada:", input_code)
    
    # Llamamos al analizador léxico para obtener los tokens
    tokens = lexical_analyzer(input_code)
    
    # Mostramos los tokens reconocidos
    print("Tokens reconocidos:")
    for token in tokens:
        print(token)
