# Importamos los tipos necesarios para las anotaciones de tipo
from typing import Dict, Set  # Dict se usa para definir diccionarios y Set para conjuntos

# Definimos la clase que implementa el sistema de razonamiento no monótonico
class NonMonotonicSystem:
    def __init__(self):
        """
        Constructor de la clase. Inicializa las reglas generales del sistema y el conjunto de hechos conocidos.
        """
        # Diccionario que contiene las reglas generales del sistema.
        # Cada regla tiene un valor por defecto ("default") y una lista de excepciones ("exceptions").
        self.rules = {
            "vuela": {"default": True, "exceptions": ["pinguino", "avestruz"]},  # Por defecto, los pájaros vuelan, excepto pingüinos y avestruces
            "nada": {"default": False, "exceptions": ["pinguino", "pez"]}       # Por defecto, los pájaros no nadan, excepto pingüinos y peces
        }
        # Conjunto para almacenar los hechos conocidos (entidades específicas que se han agregado al sistema).
        # Usamos un conjunto (Set) porque no permite duplicados y es eficiente para búsquedas.
        self.facts: Set[str] = set()

    def add_fact(self, fact: str):
        """
        Método para agregar un hecho al sistema.
        :param fact: El hecho que se desea agregar (por ejemplo, "pinguino").
        """
        # Añadimos el hecho al conjunto de hechos conocidos.
        # Esto permite que el sistema tenga en cuenta esta información en las consultas.
        self.facts.add(fact)

    def query(self, attribute: str, entity: str) -> bool:
        """
        Método para consultar si un atributo (como 'vuela' o 'nada') aplica a una entidad específica.
        :param attribute: El atributo a consultar (por ejemplo, 'vuela' o 'nada').
        :param entity: La entidad sobre la que se consulta (por ejemplo, 'pajaro', 'pinguino').
        :return: True si el atributo aplica, False en caso contrario.
        """
        # Obtenemos la regla asociada al atributo solicitado.
        # Si el atributo no existe en las reglas, devolvemos un diccionario vacío.
        rule = self.rules.get(attribute, {})
        
        # Iteramos sobre las excepciones definidas en la regla.
        # Las excepciones son entidades que no siguen el valor por defecto de la regla.
        for exception in rule.get("exceptions", []):
            # Verificamos si la entidad consultada es una excepción.
            # Esto puede ocurrir si la entidad está en los hechos conocidos o si coincide directamente con la excepción.
            if exception in self.facts or exception == entity:
                # Si la entidad es una excepción, devolvemos el valor contrario al valor por defecto de la regla.
                return not rule["default"]
        
        # Si no hay excepciones que apliquen, devolvemos el valor por defecto de la regla.
        return rule.get("default", False)

# Punto de entrada principal del programa.
if __name__ == "__main__":
    # Creamos una instancia del sistema de razonamiento no monótonico.
    # Este sistema permite manejar reglas con valores por defecto y excepciones.
    system = NonMonotonicSystem()
    
    # Caso 1: Consultamos si un pájaro genérico vuela.
    # Como no se especifica ninguna excepción, se aplica la regla por defecto.
    print("¿Pájaro genérico vuela?", system.query("vuela", "pajaro"))  # True, porque por defecto los pájaros vuelan
    
    # Caso 2: Agregamos el hecho de que estamos hablando de un pingüino.
    # Esto añade "pinguino" al conjunto de hechos conocidos.
    system.add_fact("pinguino")
    
    # Consultamos si un pingüino vuela.
    # Los pingüinos son una excepción a la regla de que los pájaros vuelan, por lo que el resultado será False.
    print("¿Pingüino vuela?", system.query("vuela", "pinguino"))  # False
    
    # Caso 3: Consultamos si un pingüino nada.
    # Los pingüinos son una excepción a la regla de que los pájaros no nadan, por lo que el resultado será True.
    print("¿Pingüino nada?", system.query("nada", "pinguino"))  # True