from lark import Lark, Transformer, v_args  # Para definir gramáticas, analizar cadenas, transformar árboles sintácticos y trabajar con análisis sintáctico estructurado.

# ---------------------------------------
# 1. Gramática formal: expresiones lógicas y aritméticas
# ---------------------------------------
# Definición de la gramática para analizar expresiones lógicas.
# La gramática incluye operadores lógicos (AND, OR, NOT) y comparaciones (==, !=, <, >, <=, >=).
grammar = """
?start: expr  // Punto de entrada de la gramática

?expr: expr "AND" term    -> and_op  // Operador lógico AND
     | expr "OR" term     -> or_op   // Operador lógico OR
     | term               // Un término puede ser una expresión

?term: "NOT" term         -> not_op  // Operador lógico NOT
     | "(" expr ")"       // Expresión entre paréntesis
     | comparison         // Comparación entre variables

?comparison: var OP var   -> comparison  // Comparación entre dos variables

var: CNAME | NUMBER       // Una variable puede ser un nombre o un número

OP: "==" | "!=" | "<" | ">" | "<=" | ">="  // Operadores de comparación

%import common.CNAME      // Importa nombres comunes (identificadores)
%import common.NUMBER     // Importa números
%import common.WS         // Importa espacios en blanco
%ignore WS                // Ignora espacios en blanco
"""

# ---------------------------------------
# 2. Transformador semántico lógico
# ---------------------------------------
# Clase que transforma el árbol sintáctico en una representación lógica más legible.
@v_args(inline=True)
class LogicTransformer(Transformer):
    # Convierte una variable en un entero si es posible, de lo contrario, la deja como cadena.
    def var(self, v):
        try:
            return int(v)
        except:
            return str(v)

    # Transforma una comparación en una representación lógica.
    def comparison(self, a, op, b):
        return f"({a} {op} {b})"

    # Transforma una operación lógica AND.
    def and_op(self, a, b):
        return f"({a} AND {b})"

    # Transforma una operación lógica OR.
    def or_op(self, a, b):
        return f"({a} OR {b})"

    # Transforma una operación lógica NOT.
    def not_op(self, a):
        return f"(NOT {a})"

# ---------------------------------------
# 3. Parser y Evaluación
# ---------------------------------------
# Crea un parser con la gramática definida y el transformador lógico.
parser = Lark(grammar, parser="lalr", transformer=LogicTransformer())

# Crea un parser sin transformador para mostrar el árbol sintáctico sin procesar.
raw_parser = Lark(grammar, parser="lalr")

# Ejemplo lógico a analizar
input_sentence = "NOT (x == 5 OR y != 3) AND z > 1"

# Parseo estructurado: transforma la entrada lógica en una estructura lógica interpretada.
logical_structure = parser.parse(input_sentence)

# Parseo sin transformar: genera el árbol sintáctico (parse tree).
parse_tree = raw_parser.parse(input_sentence)

# ---------------------------------------
# 4. Resultados
# ---------------------------------------
# Imprime la entrada lógica original.
print("Entrada lógica:")
print(input_sentence)

# Imprime la estructura lógica interpretada (resultado del transformador).
print("\nEstructura lógica interpretada:")
print(logical_structure)

# Imprime el árbol sintáctico (parse tree) en formato legible.
print("\nÁrbol sintáctico (parse tree):")
print(parse_tree.pretty())
