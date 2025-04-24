import networkx as nx
import matplotlib.pyplot as plt

# Definimos la taxonomía como relaciones de jerarquía: padre -> hijos
# Cada clave representa una categoría padre, y su valor es una lista de subcategorías u objetos hijos.
taxonomia = {
    "Entidad": ["Animal", "Planta"],  # "Entidad" es la raíz de la jerarquía
    "Animal": ["Mamífero", "Ave"],   # "Animal" tiene como hijos "Mamífero" y "Ave"
    "Mamífero": ["Perro", "Gato"],   # "Mamífero" tiene como hijos "Perro" y "Gato"
    "Ave": ["Águila", "Pato"],       # "Ave" tiene como hijos "Águila" y "Pato"
    "Planta": ["Árbol", "Flor"],     # "Planta" tiene como hijos "Árbol" y "Flor"
    "Árbol": ["Encino", "Pino"],     # "Árbol" tiene como hijos "Encino" y "Pino"
    "Flor": ["Rosa", "Tulipán"]      # "Flor" tiene como hijos "Rosa" y "Tulipán"
}

# Construimos un grafo dirigido para representar la jerarquía
# Un grafo dirigido (DiGraph) permite modelar relaciones padre-hijo con direcciones.
G = nx.DiGraph()

# Llenamos el grafo con las relaciones padre-hijo definidas en la taxonomía
for categoria, subcategorias in taxonomia.items():
    for sub in subcategorias:
        G.add_edge(categoria, sub)  # Agregamos una arista dirigida de 'categoria' a 'sub'

# Definimos los colores de los nodos: categorías en azul, objetos en verde
# Creamos una lista para almacenar los colores de los nodos según su tipo
nodos = G.nodes()  # Obtenemos todos los nodos del grafo
colores = []
for nodo in nodos:
    if nodo in taxonomia:  # Si el nodo está en la taxonomía, es una categoría
        colores.append("skyblue")  # Categoría: color azul claro
    else:  # Si no está en la taxonomía, es un objeto o instancia
        colores.append("lightgreen")  # Objeto: color verde claro

# Dibujamos el grafo jerárquico
plt.figure(figsize=(12, 8))  # Configuramos el tamaño de la figura
pos = nx.spring_layout(G, seed=42)  # Calculamos las posiciones de los nodos para el grafo
nx.draw(
    G, pos, with_labels=True, node_color=colores, node_size=2000, 
    font_size=10, edge_color='gray', arrows=True
)  # Dibujamos el grafo con etiquetas, colores y flechas
plt.title("Taxonomía de Categorías y Objetos", fontsize=14)  # Título del grafo
plt.show()  # Mostramos el grafo

# Función para encontrar la raíz de una categoría u objeto
# La raíz es el nodo más alto en la jerarquía (sin padres).
def encontrar_raiz(grafo, nodo):
    padres = list(grafo.predecessors(nodo))  # Obtenemos los nodos padres del nodo actual
    if not padres:  # Si no tiene padres, es la raíz
        return nodo
    return encontrar_raiz(grafo, padres[0])  # Llamada recursiva para buscar la raíz

# Función para clasificar un nodo
# Determina si un nodo es una categoría u objeto y encuentra su categoría raíz.
def clasificar_nodo(nodo):
    if nodo not in G:  # Si el nodo no está en el grafo, no pertenece a la taxonomía
        return f"'{nodo}' no se encuentra en la taxonomía."
    
    # Determinamos si el nodo es una categoría o un objeto
    tipo = "Categoría" if nodo in taxonomia else "Objeto"
    raiz = encontrar_raiz(G, nodo)  # Encontramos la raíz del nodo
    return f"{nodo} es un(a) {tipo}, y pertenece a la categoría raíz: '{raiz}'."

# Pruebas de clasificación
# Verificamos varios nodos para determinar su tipo y categoría raíz
print("\nCLASIFICACIÓN DE NODOS:")
nodos_a_verificar = ["Perro", "Gato", "Ave", "Rosa", "Entidad"]  # Lista de nodos a clasificar
for n in nodos_a_verificar:
    print("- " + clasificar_nodo(n))  # Imprimimos la clasificación de cada nodo
