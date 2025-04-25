from typing import Dict, List, Optional

# --- Red Semántica (Grafo de Conocimiento) ---
class SemanticNetwork:
    """
    Representa una red semántica como un grafo de conocimiento.
    Cada entidad tiene relaciones ('is_a', 'has') con otras entidades o atributos.
    """
    def __init__(self):
        # Grafo que define las relaciones entre entidades y sus atributos
        self.graph: Dict[str, Dict[str, List[str]]] = {
            # Ejemplo: "Perro" es un "Mamífero", "Mamífero" es un "Animal", etc.
            "Perro": {"is_a": ["Mamífero"], "has": ["Pelaje", "CuatroPatas"]},
            "Mamífero": {"is_a": ["Animal"], "has": ["SangreCaliente"]},
            "Pez": {"is_a": ["Animal"], "has": ["Escamas", "Nada"]},
            "Animal": {"is_a": []}  # Nodo raíz, no tiene relaciones 'is_a'
        }

    def query(self, entity: str, relation: str) -> Optional[List[str]]:
        """
        Consulta la red semántica para obtener las entidades o atributos relacionados.
        Args:
            entity (str): La entidad a consultar (e.g., "Perro").
            relation (str): La relación a buscar (e.g., "is_a", "has").
        Returns:
            Optional[List[str]]: Lista de entidades o atributos relacionados, o None si no existe.
        """
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
    # Reglas de clasificación: cada regla asocia un rasgo con una condición lógica
    rules = [
        ("TienePelaje", lambda a: "Pelaje" in network.query(a, "has")),  # Tiene pelaje si "Pelaje" está en 'has'
        ("PuedeNadar", lambda a: "Nada" in network.query(a, "has")),    # Puede nadar si "Nada" está en 'has'
        ("EsMamífero", lambda a: "Mamífero" in network.query(a, "is_a"))  # Es mamífero si "Mamífero" está en 'is_a'
    ]
    
    traits = []  # Lista para almacenar los rasgos que cumple el animal
    for trait, rule in rules:
        if rule(animal):  # Si la regla se cumple, añade el rasgo a la lista
            traits.append(trait)
    
    # Devuelve una descripción del animal con los rasgos encontrados o indica que no es clasificable
    return f"{animal} es {' y '.join(traits) or 'no clasificable'}"

# --- Ejemplo de Uso ---
if __name__ == "__main__":
    # Crear una instancia de la red semántica
    network = SemanticNetwork()
    
    # Consulta la red semántica
    print("Red Semántica:")
    print("- Un perro tiene:", network.query("Perro", "has"))  # Consulta los atributos de "Perro"
    print("- Un pez es un:", network.query("Pez", "is_a"))    # Consulta las categorías de "Pez"
    
    # Clasificación con reglas lógicas
    print("\nClasificación Lógica:")
    print(classify_animal("Perro", network))  # Clasifica "Perro" según las reglas
    print(classify_animal("Pez", network))    # Clasifica "Pez" según las reglas