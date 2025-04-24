from rdflib import Graph, Literal, RDF, RDFS, Namespace
import networkx as nx
import matplotlib.pyplot as plt

# Crear espacio de nombres para los términos de la ontología
# Este espacio de nombres define un prefijo "ex" para los recursos relacionados con animales
ex = Namespace("http://ejemplo.org/animales#")

# Crear el grafo RDF donde se almacenará la ontología
g = Graph()
g.bind("ex", ex)  # Vincular el prefijo "ex" al espacio de nombres para facilitar su uso

# Definir clases y subclases en la ontología
# Se crean las clases principales: Animal, Mamifero y Ave
g.add((ex.Animal, RDF.type, RDFS.Class))  # Animal es una clase
g.add((ex.Mamifero, RDF.type, RDFS.Class))  # Mamifero es una clase
g.add((ex.Ave, RDF.type, RDFS.Class))  # Ave es una clase

# Definir relaciones jerárquicas entre clases
# Mamifero y Ave son subclases de Animal
g.add((ex.Mamifero, RDFS.subClassOf, ex.Animal))
g.add((ex.Ave, RDFS.subClassOf, ex.Animal))

# Definir individuos (instancias) de las clases
# Leon es un Mamifero, y Pinguino es un Ave
g.add((ex.Leon, RDF.type, ex.Mamifero))
g.add((ex.Pinguino, RDF.type, ex.Ave))

# Definir propiedades y sus relaciones con individuos
# Se define la propiedad "habitaEn" y se asignan valores a los individuos
g.add((ex.habitaEn, RDF.type, RDF.Property))  # habitaEn es una propiedad
g.add((ex.Leon, ex.habitaEn, Literal("Sabana")))  # Leon habita en la Sabana
g.add((ex.Pinguino, ex.habitaEn, Literal("Antártida")))  # Pinguino habita en la Antártida

# Crear un grafo dirigido para la visualización utilizando NetworkX
G_vis = nx.DiGraph()

# Añadir nodos y aristas al grafo de visualización a partir del grafo RDF
# Cada triple (sujeto, predicado, objeto) se convierte en un nodo o arista
for s, p, o in g:
    # Extraer etiquetas legibles para los nodos y aristas
    s_label = s.split("#")[-1] if "#" in str(s) else str(s)
    p_label = p.split("#")[-1] if "#" in str(p) else str(p)
    o_label = o.split("#")[-1] if "#" in str(o) else str(o)
    # Añadir una arista al grafo con etiquetas
    G_vis.add_edge(s_label, o_label, label=p_label)

# Dibujar el grafo utilizando NetworkX y Matplotlib
# Se utiliza un diseño de grafo con disposición por resorte
pos = nx.spring_layout(G_vis, seed=42)  # Posiciones de los nodos
edge_labels = nx.get_edge_attributes(G_vis, 'label')  # Etiquetas de las aristas

# Configurar el tamaño y estilo del grafo
plt.figure(figsize=(10, 6))
nx.draw(G_vis, pos, with_labels=True, node_color='lightblue', node_size=2000, font_size=10, arrows=True)
nx.draw_networkx_edge_labels(G_vis, pos, edge_labels=edge_labels, font_size=9)
plt.title("Visualización de Ontología: Animales", fontsize=14)
plt.tight_layout()
plt.show()  # Mostrar el grafo
