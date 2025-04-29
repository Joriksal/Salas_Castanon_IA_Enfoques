import math
from collections import Counter

class Nodo:
    """Clase para representar un nodo en el árbol de decisión."""
    def __init__(self, atributo=None, valor=None, resultado=None):
        self.atributo = atributo  # Atributo usado para dividir (ej: "Clima")
        self.valor = valor        # Valor del atributo (ej: "Soleado")
        self.ramas = {}           # Diccionario de ramas {valor: Nodo}
        self.resultado = resultado  # Resultado si es hoja (ej: "Jugar")

def entropia(datos):
    """
    Calcula la entropía de un conjunto de datos.
    La entropía mide la incertidumbre en la distribución de clases.
    """
    conteo = Counter(dato[-1] for dato in datos)  # Cuenta las clases en los datos
    total = len(datos)  # Total de ejemplos
    # Fórmula de entropía: -sum(p * log2(p)) para cada clase
    return -sum((count / total) * math.log2(count / total) for count in conteo.values())

def ganancia_informacion(datos, atributo_idx):
    """
    Calcula la ganancia de información al dividir los datos por un atributo.
    La ganancia de información mide cuánto reduce la entropía al dividir.
    """
    entropia_total = entropia(datos)  # Entropía antes de dividir
    valores = set(dato[atributo_idx] for dato in datos)  # Valores únicos del atributo
    nueva_entropia = 0.0
    for valor in valores:
        # Filtra los datos que tienen el valor actual del atributo
        subconjunto = [dato for dato in datos if dato[atributo_idx] == valor]
        peso = len(subconjunto) / len(datos)  # Proporción del subconjunto
        nueva_entropia += peso * entropia(subconjunto)  # Entropía ponderada
    # Ganancia de información: entropía inicial - entropía después de dividir
    return entropia_total - nueva_entropia

def id3(datos, atributos):
    """
    Implementa el algoritmo ID3 para construir un árbol de decisión.
    - datos: Lista de ejemplos de entrenamiento.
    - atributos: Lista de nombres de atributos disponibles.
    """
    clases = [dato[-1] for dato in datos]  # Extrae las clases de los datos
    # Caso base: Si todos los ejemplos tienen la misma clase, retorna un nodo hoja
    if len(set(clases)) == 1:
        return Nodo(resultado=clases[0])

    # Caso base: Si no hay más atributos para dividir, retorna la clase más común
    if not atributos:
        clase_mas_comun = Counter(clases).most_common(1)[0][0]
        return Nodo(resultado=clase_mas_comun)

    # Selecciona el mejor atributo para dividir (el que maximiza la ganancia de información)
    mejor_atributo_nombre = max(
        atributos, 
        key=lambda attr: ganancia_informacion(datos, atributos.index(attr))
    )
    mejor_atributo_idx = atributos.index(mejor_atributo_nombre)

    # Crea un nodo para el mejor atributo
    nodo = Nodo(atributo=mejor_atributo_nombre)
    valores = set(dato[mejor_atributo_idx] for dato in datos)  # Valores únicos del atributo

    for valor in valores:
        # Crea un subconjunto de datos eliminando el atributo seleccionado
        subconjunto = [
            dato[:mejor_atributo_idx] + dato[mejor_atributo_idx+1:] 
            for dato in datos 
            if dato[mejor_atributo_idx] == valor
        ]
        # Actualiza la lista de atributos eliminando el atributo usado
        nuevos_atributos = [attr for attr in atributos if attr != mejor_atributo_nombre]
        # Construye recursivamente el árbol para el subconjunto
        nodo.ramas[valor] = id3(subconjunto, nuevos_atributos)

    return nodo

def imprimir_arbol(nodo, sangria=""):
    """
    Imprime el árbol de decisión de forma recursiva.
    - nodo: Nodo actual del árbol.
    - sangria: Espaciado para mostrar la jerarquía del árbol.
    """
    if nodo.resultado is not None:
        # Si es un nodo hoja, imprime el resultado
        print(sangria + "Resultado:", nodo.resultado)
        return
    # Si es un nodo interno, imprime el atributo y recorre las ramas
    print(sangria + f"Atributo: {nodo.atributo}")
    for valor, rama in nodo.ramas.items():
        print(sangria + f"--> Valor: {valor}")
        imprimir_arbol(rama, sangria + "   ")

# --- Ejemplo práctico ---
if __name__ == "__main__":
    # Datos de entrenamiento: [Clima, Temperatura, Humedad, Viento, Jugar?]
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
    atributos = ["Clima", "Temperatura", "Humedad", "Viento"]

    # Construir el árbol
    arbol = id3(datos, atributos)
    print("Árbol de Decisión ID3:")
    imprimir_arbol(arbol)