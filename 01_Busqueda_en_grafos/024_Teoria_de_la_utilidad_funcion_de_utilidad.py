class AgenteUtilidad:
    def __init__(self, opciones, funcion_utilidad=None):
        """
        Inicializa un agente de decisión basado en utilidad.
        
        Args:
            opciones: Diccionario de opciones con sus atributos.
            funcion_utilidad: Función personalizada para calcular utilidad (opcional).
        """
        self.opciones = opciones  # Almacena las opciones disponibles.
        # Si no se proporciona una función de utilidad personalizada, usa la predeterminada.
        self.funcion_utilidad = funcion_utilidad or self.funcion_utilidad_default
        
        # Diccionario de pesos para los atributos. Los pesos determinan la importancia relativa
        # de cada atributo en el cálculo de la utilidad.
        self.pesos = {
            'beneficio': 0.6,  # Peso positivo porque se busca maximizar el beneficio.
            'costo': -0.3,     # Peso negativo porque se busca minimizar el costo.
            'riesgo': -0.1     # Peso negativo porque se busca minimizar el riesgo.
        }
    
    def funcion_utilidad_default(self, opcion):
        """
        Calcula la utilidad de una opción usando una función lineal ponderada.
        
        Args:
            opcion: Diccionario con los atributos de la opción.
        
        Returns:
            float: Valor de la utilidad calculada.
        """
        utilidad = 0  # Inicializa la utilidad en 0.
        # Itera sobre los atributos de la opción y calcula la suma ponderada.
        for atributo, valor in opcion.items():
            if atributo in self.pesos:  # Verifica si el atributo tiene un peso definido.
                utilidad += self.pesos[atributo] * valor  # Suma el producto del peso y el valor.
        return utilidad  # Devuelve la utilidad calculada.
    
    def evaluar_opciones(self):
        """
        Evalúa todas las opciones y devuelve sus utilidades.
        
        Returns:
            dict: Diccionario {nombre_opcion: utilidad}.
        """
        # Usa un diccionario por comprensión para calcular la utilidad de cada opción.
        return {nombre: self.funcion_utilidad(attrs) for nombre, attrs in self.opciones.items()}
    
    def mejor_opcion(self):
        """
        Selecciona la opción con mayor utilidad.
        
        Returns:
            tuple: (nombre_opcion, utilidad).
        """
        utilidades = self.evaluar_opciones()  # Calcula las utilidades de todas las opciones.
        # Encuentra la opción con la mayor utilidad usando la función max.
        return max(utilidades.items(), key=lambda x: x[1])
    
    def grafo_decision(self):
        """
        Representa las opciones como un grafo de decisiones con utilidades.
        
        Returns:
            dict: Grafo representado como un diccionario {origen: {destino: utilidad}}.
        """
        grafo = {}  # Inicializa el grafo como un diccionario vacío.
        utilidades = self.evaluar_opciones()  # Calcula las utilidades de todas las opciones.
        
        # Nodo inicial "Inicio" conectado a cada opción con su utilidad.
        for opcion, attrs in self.opciones.items():
            grafo.setdefault('Inicio', {})[opcion] = utilidades[opcion]
            
            # Aquí podríamos agregar más nodos para decisiones secuenciales.
            # Por simplicidad, mostramos solo la primera decisión.
            
        return grafo  # Devuelve el grafo de decisiones.

# --------------------------------------------
# Ejemplo de uso: Decisión de inversión
# --------------------------------------------

if __name__ == "__main__":
    # Definir las opciones de inversión con sus atributos.
    opciones_inversion = {
        'Acciones': {
            'beneficio': 8,  # Escala de 1-10.
            'costo': 6,      # Escala de 1-10 (10 = mayor costo).
            'riesgo': 7      # Escala de 1-10.
        },
        'Bonos': {
            'beneficio': 5,
            'costo': 3,
            'riesgo': 2
        },
        'Bienes_Raices': {
            'beneficio': 7,
            'costo': 8,
            'riesgo': 4
        },
        'Criptomonedas': {
            'beneficio': 9,
            'costo': 5,
            'riesgo': 9
        }
    }
    
    # Crear el agente de decisión.
    agente = AgenteUtilidad(opciones_inversion)
    
    # Evaluar todas las opciones.
    print("Utilidades calculadas:")
    utilidades = agente.evaluar_opciones()
    for opcion, utilidad in utilidades.items():
        print(f"{opcion}: {utilidad:.2f}")
    
    # Seleccionar la mejor opción.
    mejor, utilidad = agente.mejor_opcion()
    print(f"\nMejor opción: {mejor} con utilidad {utilidad:.2f}")
    
    # Mostrar grafo de decisión.
    print("\nGrafo de decisión:")
    grafo = agente.grafo_decision()
    for origen, destinos in grafo.items():
        print(f"{origen} ->")
        for destino, utilidad in destinos.items():
            print(f"  {destino} (utilidad: {utilidad:.2f})")