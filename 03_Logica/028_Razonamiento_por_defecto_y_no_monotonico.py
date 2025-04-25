from typing import Dict, Set

class NonMonotonicSystem:
    def __init__(self):
        # Definimos las reglas generales con sus valores por defecto y excepciones
        self.rules = {
            "vuela": {"default": True, "exceptions": ["pinguino", "avestruz"]},  # Por defecto, los pájaros vuelan, excepto pingüinos y avestruces
            "nada": {"default": False, "exceptions": ["pinguino", "pez"]}       # Por defecto, los pájaros no nadan, excepto pingüinos y peces
        }
        self.facts: Set[str] = set()  # Conjunto de hechos conocidos (entidades específicas)

    def add_fact(self, fact: str):
        """Agrega un hecho al sistema."""
        self.facts.add(fact)  # Añadimos el hecho al conjunto de hechos conocidos

    def query(self, attribute: str, entity: str) -> bool:
        """
        Consulta si un atributo (como 'vuela' o 'nada') aplica a una entidad específica.
        :param attribute: El atributo a consultar (por ejemplo, 'vuela' o 'nada').
        :param entity: La entidad sobre la que se consulta (por ejemplo, 'pajaro', 'pinguino').
        :return: True si el atributo aplica, False en caso contrario.
        """
        # Obtenemos la regla asociada al atributo
        rule = self.rules.get(attribute, {})
        
        # Verificamos si la entidad está en las excepciones de la regla
        for exception in rule.get("exceptions", []):
            if exception in self.facts or exception == entity:
                # Si la entidad es una excepción, devolvemos el valor contrario al default
                return not rule["default"]
        
        # Si no hay excepciones, devolvemos el valor por defecto de la regla
        return rule.get("default", False)

if __name__ == "__main__":
    # Creamos una instancia del sistema de razonamiento no monótonico
    system = NonMonotonicSystem()
    
    # Caso 1: Consultamos si un pájaro genérico vuela (regla por defecto)
    print("¿Pájaro genérico vuela?", system.query("vuela", "pajaro"))  # True, porque por defecto los pájaros vuelan
    
    # Caso 2: Agregamos el hecho de que estamos hablando de un pingüino
    system.add_fact("pinguino")
    print("¿Pingüino vuela?", system.query("vuela", "pinguino"))  # False, porque los pingüinos son una excepción a la regla de que los pájaros vuelan
    
    # Caso 3: Consultamos si un pingüino nada (excepción a la regla de que los pájaros no nadan)
    print("¿Pingüino nada?", system.query("nada", "pinguino"))  # True, porque los pingüinos son una excepción y sí nadan