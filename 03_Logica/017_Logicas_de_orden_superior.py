# Importamos el módulo `typing` para usar anotaciones de tipo como `Callable` y `Any`
from typing import Callable, Any

class MathAssistant:
    """Sistema de verificación matemática basado en Lógica de Orden Superior.
    
    Este sistema permite definir axiomas, verificar propiedades matemáticas
    y trabajar con cuantificadores universales y existenciales personalizados.
    """

    def __init__(self):
        """Inicializa el asistente matemático con axiomas básicos y operaciones personalizadas."""
        # Diccionario para almacenar axiomas y teoremas definidos en el sistema
        self.theorems = {}
        
        # Operaciones personalizadas que incluyen cuantificadores y lógica condicional
        self.custom_ops = {
            # Implementación de la implicación lógica: "si x entonces y"
            'implies': lambda x, y: (not x) or y,
            
            # Cuantificador universal: verifica si una propiedad `f` se cumple
            # para todos los valores en el rango [start, end]
            'forall': lambda f, start=0, end=100: all(f(n) for n in range(start, end+1)),
            
            # Cuantificador existencial: verifica si existe al menos un valor
            # en el rango [start, end] que cumpla la propiedad `f`
            'exists': lambda f, start=0, end=100: any(f(n) for n in range(start, end+1))
        }
        
        # Definimos axiomas básicos del sistema
        self.add_axiom(
            self._create_axiom(
                "forall(P: Callable[[int], bool], P(0) and forall(n, implies(P(n), P(n+1))) implies forall(n, P(n))"
            ),
            name="Inducción"  # Principio de inducción matemática
        )
        
        self.add_axiom(
            self._create_axiom(
                "forall(f: Callable, x, y, implies(x == y, f(x) == f(y)))"
            ),
            name="Leibniz"  # Principio de identidad de Leibniz
        )

    def _create_axiom(self, expr: str) -> Callable:
        """Crea una función de axioma a partir de una expresión lógica.

        Args:
            expr (str): Expresión lógica que define el axioma.

        Returns:
            Callable: Función que implementa el axioma.
        """
        # Verificamos si la expresión corresponde al principio de inducción matemática
        if "implies(P(n), P(n+1))" in expr:
            return lambda P: (
                P(0) and  # La propiedad debe cumplirse para n = 0
                all(self.custom_ops['implies'](P(n), P(n+1)) for n in range(100))  # Inducción
            ) and all(P(n) for n in range(101))  # Verificación para todos los n
        
        # Verificamos si la expresión corresponde al principio de identidad de Leibniz
        elif "f(x) == f(y)" in expr:
            return lambda f, x, y: (x == y) or (f(x) == f(y))
        
        # Si la expresión no coincide con ningún axioma conocido, lanzamos un error
        raise ValueError(f"No se pudo compilar la expresión: {expr}")

    def add_axiom(self, axiom_func: Callable, name: str):
        """Añade un axioma al sistema.

        Args:
            axiom_func (Callable): Función que implementa el axioma.
            name (str): Nombre del axioma.
        """
        self.theorems[name] = {'type': 'axiom', 'expr': axiom_func}

    def check_property(self, prop_func: Callable, *args) -> bool:
        """Verifica si una propiedad matemática se cumple.

        Args:
            prop_func (Callable): Función que representa la propiedad a verificar.
            *args: Argumentos necesarios para evaluar la propiedad.

        Returns:
            bool: True si la propiedad se cumple, False en caso contrario.
        """
        try:
            return prop_func(*args)
        except:
            return False

    # ----------- Teoría de Grupos -----------

    def define_group_axioms(self):
        """Define los axiomas básicos de la teoría de grupos."""
        # Axioma de asociatividad: verifica si una operación binaria es asociativa
        self.add_axiom(
            lambda op: all(
                op(op(a, b), c) == op(a, op(b, c))  # Verifica la propiedad asociativa
                for a in range(4) for b in range(4) for c in range(4)
            ),
            name="Asociatividad"
        )
        
        # Axioma de identidad: verifica si existe un elemento neutro para la operación
        self.add_axiom(
            lambda op, e: all(
                op(e, a) == a and op(a, e) == a  # Verifica la existencia del elemento identidad
                for a in range(4)
            ),
            name="Identidad"
        )

# -------------------------------
# Ejemplo de Uso Correcto
# -------------------------------
if __name__ == "__main__":
    print("=== Sistema de Verificación Matemática ===")
    
    # Crear una instancia del asistente matemático
    ma = MathAssistant()
    
    # 1. Verificar principio de inducción
    def propiedad_ejemplo(n: int) -> bool:
        """Propiedad de ejemplo: todos los números naturales son mayores o iguales a 0."""
        return n >= 0
    
    print("\n1. Verificación del principio de inducción:")
    axioma_induccion = ma.theorems["Inducción"]['expr']
    print("¿Se cumple para propiedad_ejemplo?", 
          axioma_induccion(propiedad_ejemplo))  # Debería ser True
    
    # 2. Teoría de grupos
    print("\n2. Teoría de Grupos:")
    ma.define_group_axioms()
    
    # Operación de grupo módulo 4
    def suma_mod4(a, b):
        """Suma módulo 4: operación binaria definida como (a + b) % 4."""
        return (a + b) % 4
    
    # Verificar axiomas de teoría de grupos
    axioma_asociatividad = ma.theorems["Asociatividad"]['expr']
    axioma_identidad = ma.theorems["Identidad"]['expr']
    
    print("¿Es asociativa suma_mod4?", axioma_asociatividad(suma_mod4))  # True
    print("¿Tiene elemento identidad?", axioma_identidad(suma_mod4, 0))  # True
    
    # 3. Verificación de propiedad universal
    print("\n3. Verificación de propiedad universal:")
    def cuadrado_no_negativo(x):
        """Propiedad: el cuadrado de cualquier número es no negativo."""
        return x**2 >= 0
    
    # Usando el cuantificador forall personalizado
    print("¿x² ≥ 0 para x ∈ [-100, 100]?", 
          ma.custom_ops['forall'](cuadrado_no_negativo, -100, 100))  # True