# Importamos tipos específicos de anotaciones para mejorar la legibilidad y el tipado del código
from typing import Dict, List, Optional

# --- Red Semántica (Grafo de Conocimiento) ---
class SemanticNetwork:
    """
    Representa una red semántica como un grafo de conocimiento.
    Cada entidad tiene relaciones ('is_a', 'has') con otras entidades o atributos.
    """

    def __init__(self):
        """
        Constructor de la clase SemanticNetwork.
        Inicializa un grafo que define las relaciones entre entidades y sus atributos.
        """
        # Diccionario que representa el grafo de conocimiento.
        # Cada clave es una entidad (e.g., "Perro") y su valor es otro diccionario
        # que contiene relaciones ('is_a', 'has') y sus valores asociados.
        self.graph: Dict[str, Dict[str, List[str]]] = {
            # Ejemplo de relaciones:
            # "Perro" es un "Mamífero" y tiene "Pelaje" y "CuatroPatas".
            "Perro": {"is_a": ["Mamífero"], "has": ["Pelaje", "CuatroPatas"]},
            # "Mamífero" es un "Animal" y tiene "SangreCaliente".
            "Mamífero": {"is_a": ["Animal"], "has": ["SangreCaliente"]},
            # "Pez" es un "Animal" y tiene "Escamas" y "Nada".
            "Pez": {"is_a": ["Animal"], "has": ["Escamas", "Nada"]},
            # "Animal" es el nodo raíz y no tiene relaciones 'is_a'.
            "Animal": {"is_a": []}
        }

    def query(self, entity: str, relation: str) -> Optional[List[str]]:
        """
        Consulta la red semántica para obtener las entidades o atributos relacionados.

        Args:
            entity (str): La entidad a consultar (e.g., "Perro").
            relation (str): La relación a buscar (e.g., "is_a", "has").

        Returns:
            Optional[List[str]]: Lista de entidades o atributos relacionados, 
            o None si no existe la entidad o la relación.
        """
        # Busca en el grafo la entidad y la relación especificada.
        # Si no existe, devuelve None.
        return self.graph.get(entity, {}).get(relation, None)

# --- Lógica Descriptiva (Reglas de Clasificación) ---
def classify_animal(animal: str, network: SemanticNetwork) -> str:
    """
    Clasifica un animal usando reglas lógicas basadas en la red semántica.

    Args:
        animal (str): El nombre del animal a clasificar (e.g., "Perro").
        network (SemanticNetwork): La red semántica que contiene las relaciones.

    Returns:
        str: Una descripción de las características del animal.
    """
    # Lista de reglas de clasificación.
    # Cada regla asocia un rasgo (e.g., "TienePelaje") con una condición lógica (lambda).
    rules = [
        # Regla: "TienePelaje" si "Pelaje" está en la relación 'has' del animal.
        ("TienePelaje", lambda a: "Pelaje" in network.query(a, "has")),
        # Regla: "PuedeNadar" si "Nada" está en la relación 'has' del animal.
        ("PuedeNadar", lambda a: "Nada" in network.query(a, "has")),
        # Regla: "EsMamífero" si "Mamífero" está en la relación 'is_a' del animal.
        ("EsMamífero", lambda a: "Mamífero" in network.query(a, "is_a"))
    ]
    
    # Lista para almacenar los rasgos que cumple el animal.
    traits = []
    for trait, rule in rules:
        # Aplica cada regla al animal.
        # Si la regla se cumple, añade el rasgo a la lista.
        if rule(animal):
            traits.append(trait)
    
    # Devuelve una descripción del animal con los rasgos encontrados.
    # Si no se encuentra ningún rasgo, indica que el animal "no es clasificable".
    return f"{animal} es {' y '.join(traits) or 'no clasificable'}"

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Crear una instancia de la red semántica.
    network = SemanticNetwork()
    
    # Consulta la red semántica.
    print("Red Semántica:")
    # Consulta los atributos de "Perro" usando la relación 'has'.
    print("- Un perro tiene:", network.query("Perro", "has"))
    # Consulta las categorías de "Pez" usando la relación 'is_a'.
    print("- Un pez es un:", network.query("Pez", "is_a"))
    
    # Clasificación con reglas lógicas.
    print("\nClasificación Lógica:")
    # Clasifica "Perro" según las reglas definidas.
    print(classify_animal("Perro", network))
    # Clasifica "Pez" según las reglas definidas.
    print(classify_animal("Pez", network))