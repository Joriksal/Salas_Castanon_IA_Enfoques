# Importamos la librería `math` para realizar cálculos matemáticos como logaritmos.
import math

# Importamos `Counter` de la librería `collections` para contar elementos en listas de manera eficiente.
from collections import Counter

class Nodo:
    """
    Clase para representar un nodo en el árbol de decisión.
    Cada nodo puede ser un nodo interno (con ramas hacia otros nodos) o un nodo hoja (con un resultado).
    """
    def __init__(self, atributo=None, valor=None, resultado=None):
        """
        Inicializa un nodo del árbol.
        - atributo: El atributo usado para dividir en este nodo (ej: "Clima").
        - valor: El valor del atributo que llevó a este nodo (ej: "Soleado").
        - resultado: El resultado si este nodo es una hoja (ej: "Jugar").
        """
        self.atributo = atributo  # Nombre del atributo usado para dividir.
        self.valor = valor        # Valor del atributo que llevó a este nodo.
        self.ramas = {}           # Diccionario que almacena las ramas {valor: Nodo}.
        self.resultado = resultado  # Resultado si es un nodo hoja.

def entropia(datos):
    """
    Calcula la entropía de un conjunto de datos.
    La entropía mide la incertidumbre o desorden en la distribución de clases.
    - datos: Lista de ejemplos, donde cada ejemplo es una lista y el último elemento es la clase.
    """
    # Cuenta cuántos ejemplos pertenecen a cada clase (último elemento de cada ejemplo).
    conteo = Counter(dato[-1] for dato in datos)
    total = len(datos)  # Total de ejemplos en el conjunto de datos.

    # Fórmula de entropía: -sum(p * log2(p)) para cada clase.
    return -sum((count / total) * math.log2(count / total) for count in conteo.values())

def ganancia_informacion(datos, atributo_idx):
    """
    Calcula la ganancia de información al dividir los datos por un atributo específico.
    La ganancia de información mide cuánto reduce la entropía al dividir los datos.
    - datos: Lista de ejemplos de entrenamiento.
    - atributo_idx: Índice del atributo en los ejemplos.
    """
    # Calcula la entropía total antes de dividir.
    entropia_total = entropia(datos)

    # Obtiene los valores únicos del atributo en el índice dado.
    valores = set(dato[atributo_idx] for dato in datos)

    # Inicializa la nueva entropía después de dividir.
    nueva_entropia = 0.0

    # Itera sobre cada valor único del atributo.
    for valor in valores:
        # Filtra los datos que tienen el valor actual del atributo.
        subconjunto = [dato for dato in datos if dato[atributo_idx] == valor]

        # Calcula el peso del subconjunto (proporción del subconjunto respecto al total).
        peso = len(subconjunto) / len(datos)

        # Suma la entropía ponderada del subconjunto.
        nueva_entropia += peso * entropia(subconjunto)

    # La ganancia de información es la diferencia entre la entropía inicial y la nueva entropía.
    return entropia_total - nueva_entropia

def id3(datos, atributos):
    """
    Implementa el algoritmo ID3 para construir un árbol de decisión.
    - datos: Lista de ejemplos de entrenamiento.
    - atributos: Lista de nombres de atributos disponibles para dividir.
    """
    # Extrae las clases (último elemento de cada ejemplo) de los datos.
    clases = [dato[-1] for dato in datos]

    # Caso base 1: Si todos los ejemplos tienen la misma clase, retorna un nodo hoja con esa clase.
    if len(set(clases)) == 1:
        return Nodo(resultado=clases[0])

    # Caso base 2: Si no hay más atributos para dividir, retorna un nodo hoja con la clase más común.
    if not atributos:
        clase_mas_comun = Counter(clases).most_common(1)[0][0]
        return Nodo(resultado=clase_mas_comun)

    # Selecciona el mejor atributo para dividir (el que maximiza la ganancia de información).
    mejor_atributo_nombre = max(
        atributos, 
        key=lambda attr: ganancia_informacion(datos, atributos.index(attr))
    )
    mejor_atributo_idx = atributos.index(mejor_atributo_nombre)

    # Crea un nodo para el mejor atributo seleccionado.
    nodo = Nodo(atributo=mejor_atributo_nombre)

    # Obtiene los valores únicos del mejor atributo.
    valores = set(dato[mejor_atributo_idx] for dato in datos)

    # Itera sobre cada valor único del atributo.
    for valor in valores:
        # Crea un subconjunto de datos eliminando el atributo seleccionado.
        subconjunto = [
            dato[:mejor_atributo_idx] + dato[mejor_atributo_idx+1:] 
            for dato in datos 
            if dato[mejor_atributo_idx] == valor
        ]

        # Actualiza la lista de atributos eliminando el atributo usado.
        nuevos_atributos = [attr for attr in atributos if attr != mejor_atributo_nombre]

        # Construye recursivamente el árbol para el subconjunto.
        nodo.ramas[valor] = id3(subconjunto, nuevos_atributos)

    return nodo

def imprimir_arbol(nodo, sangria=""):
    """
    Imprime el árbol de decisión de forma recursiva.
    - nodo: Nodo actual del árbol.
    - sangria: Espaciado para mostrar la jerarquía del árbol.
    """
    # Si el nodo es una hoja, imprime el resultado.
    if nodo.resultado is not None:
        print(sangria + "Resultado:", nodo.resultado)
        return

    # Si el nodo es interno, imprime el atributo y recorre las ramas.
    print(sangria + f"Atributo: {nodo.atributo}")
    for valor, rama in nodo.ramas.items():
        print(sangria + f"--> Valor: {valor}")
        imprimir_arbol(rama, sangria + "   ")

# --- Ejemplo práctico ---
if __name__ == "__main__":
    # Datos de entrenamiento: Cada ejemplo es una lista [Atributo1, Atributo2, ..., Clase].
    datos = [
        ["Soleado", "Alta", "Alta", "Débil", "No"],
        ["Soleado", "Alta", "Alta", "Fuerte", "No"],
        ["Nublado", "Alta", "Alta", "Débil", "Sí"],
        ["Lluvia", "Media", "Alta", "Débil", "Sí"],
        ["Lluvia", "Baja", "Normal", "Débil", "Sí"],
        ["Lluvia", "Baja", "Normal", "Fuerte", "No"],
        ["Nublado", "Baja", "Normal", "Fuerte", "Sí"],
        ["Soleado", "Media", "Alta", "Débil", "No"],
    ]

    # Lista de nombres de atributos (sin incluir la clase).
    atributos = ["Clima", "Temperatura", "Humedad", "Viento"]

    # Construir el árbol de decisión usando el algoritmo ID3.
    arbol = id3(datos, atributos)

    # Imprimir el árbol de decisión construido.
    print("Árbol de Decisión ID3:")
    imprimir_arbol(arbol)