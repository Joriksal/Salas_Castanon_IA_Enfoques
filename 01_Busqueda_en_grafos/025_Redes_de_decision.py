from itertools import product  # Importa una función para generar combinaciones de valores.

class Nodo:
    def __init__(self, nombre, tipo, valores=None, padres=None):
        """
        Crea un nodo para la red de decisión.
        
        Args:
            nombre: Identificador del nodo.
            tipo: 'chance', 'decision' o 'utilidad'.
            valores: Posibles valores del nodo (solo para nodos de tipo 'chance' o 'decision').
            padres: Lista de nodos padres (opcional, para dependencias).
        """
        self.nombre = nombre  # Nombre del nodo.
        self.tipo = tipo  # Tipo del nodo: 'chance', 'decision' o 'utilidad'.
        self.valores = valores if valores is not None else []  # Valores posibles del nodo.
        self.padres = padres if padres is not None else []  # Nodos padres del nodo actual.
        self.tabla = None  # Tabla de probabilidad o utilidad asociada al nodo.
        
    def __repr__(self):
        """
        Representación del nodo para depuración.
        """
        return f"{self.tipo.upper()}({self.nombre})"  # Devuelve el tipo y nombre del nodo.

class RedDecision:
    def __init__(self):
        """
        Inicializa una red de decisión vacía.
        """
        self.nodos = []  # Lista de todos los nodos en la red.
        self.nodos_decision = []  # Lista de nodos de tipo 'decision'.
        self.nodos_utilidad = []  # Lista de nodos de tipo 'utilidad'.
        
    def agregar_nodo(self, nodo):
        """
        Agrega un nodo a la red.
        
        Args:
            nodo: Nodo a agregar.
        """
        self.nodos.append(nodo)  # Añade el nodo a la lista general.
        if nodo.tipo == 'decision':  # Si el nodo es de tipo 'decision', lo añade a la lista correspondiente.
            self.nodos_decision.append(nodo)
        elif nodo.tipo == 'utilidad':  # Si el nodo es de tipo 'utilidad', lo añade a la lista correspondiente.
            self.nodos_utilidad.append(nodo)
    
    def agregar_tabla_probabilidad(self, nodo, tabla):
        """
        Asigna una tabla de probabilidad a un nodo de tipo 'chance'.
        
        Args:
            nodo: Nodo al que se asignará la tabla.
            tabla: Diccionario con las probabilidades.
        """
        if nodo.tipo != 'chance':  # Verifica que el nodo sea de tipo 'chance'.
            raise ValueError("Solo nodos de tipo 'chance' pueden tener tabla de probabilidad.")
        nodo.tabla = tabla  # Asigna la tabla de probabilidad al nodo.
    
    def agregar_tabla_utilidad(self, nodo, tabla):
        """
        Asigna una tabla de utilidad a un nodo de tipo 'utilidad'.
        
        Args:
            nodo: Nodo al que se asignará la tabla.
            tabla: Diccionario con las utilidades.
        """
        if nodo.tipo != 'utilidad':  # Verifica que el nodo sea de tipo 'utilidad'.
            raise ValueError("Solo nodos de tipo 'utilidad' pueden tener tabla de utilidad.")
        nodo.tabla = tabla  # Asigna la tabla de utilidad al nodo.
    
    def posibles_combinaciones(self, nodos):
        """
        Genera todas las combinaciones posibles de valores para un conjunto de nodos.
        
        Args:
            nodos: Lista de nodos.
        
        Returns:
            list: Lista de combinaciones posibles.
        """
        valores = [nodo.valores for nodo in nodos]  # Obtiene los valores posibles de cada nodo.
        return list(product(*valores))  # Genera todas las combinaciones posibles usando 'product'.
    
    def evaluar_decision(self, decision, evidencias={}):
        """
        Evalúa una decisión específica dado un conjunto de evidencias.
        
        Args:
            decision: Diccionario {nodo_decision: valor}.
            evidencias: Diccionario con nodos observados y sus valores.
        
        Returns:
            float: Utilidad total esperada para la decisión.
        """
        observado = {**evidencias, **decision}  # Combina las evidencias con la decisión.
        utilidad_total = 0  # Inicializa la utilidad total en 0.
        
        for nodo_utilidad in self.nodos_utilidad:  # Itera sobre los nodos de utilidad.
            padres = nodo_utilidad.padres  # Obtiene los padres del nodo de utilidad.
            valores_padres = tuple(observado.get(p.nombre, None) for p in padres)  # Obtiene los valores de los padres.
            
            if valores_padres in nodo_utilidad.tabla:  # Verifica si los valores están en la tabla.
                utilidad_total += nodo_utilidad.tabla[valores_padres]  # Suma la utilidad correspondiente.
        
        return utilidad_total  # Devuelve la utilidad total.
    
    def mejor_decision(self, evidencias={}):
        """
        Encuentra la decisión óptima maximizando la utilidad esperada.
        
        Args:
            evidencias: Diccionario con nodos observados y sus valores.
        
        Returns:
            tuple: (mejor_decisión, utilidad_esperada).
        """
        mejor_utilidad = -float('inf')  # Inicializa con un valor muy bajo.
        mejor_decision = None  # Inicializa la mejor decisión como None.
        
        decisiones_posibles = self.posibles_combinaciones(self.nodos_decision)  # Genera todas las combinaciones posibles.
        
        for decision_combo in decisiones_posibles:  # Itera sobre cada combinación de decisiones.
            decision = {d.nombre: val for d, val in zip(self.nodos_decision, decision_combo)}  # Crea un diccionario con la combinación actual.
            utilidad = self.evaluar_decision(decision, evidencias)  # Evalúa la utilidad de esta decisión.
            
            if utilidad > mejor_utilidad:  # Actualiza la mejor decisión si la utilidad es mayor.
                mejor_utilidad = utilidad
                mejor_decision = decision
        
        return mejor_decision, mejor_utilidad  # Devuelve la mejor decisión y su utilidad.

# Ejemplo: Problema del paraguas
def ejemplo_paraguas():
    """
    Crea una red de decisión para el problema del paraguas.
    
    Returns:
        RedDecision: Red de decisión configurada.
    """
    red = RedDecision()  # Crear la red de decisión.
    
    # Nodos chance (aleatorios).
    clima = Nodo('Clima', 'chance', ['soleado', 'lluvioso'])
    pronostico = Nodo('Pronostico', 'chance', ['bueno', 'malo'], padres=[clima])
    
    # Nodo de decisión.
    llevar_paraguas = Nodo('LlevarParaguas', 'decision', ['si', 'no'])
    
    # Nodo de utilidad.
    utilidad = Nodo('Utilidad', 'utilidad', padres=[clima, llevar_paraguas])
    
    # Agregar nodos a la red.
    red.agregar_nodo(clima)
    red.agregar_nodo(pronostico)
    red.agregar_nodo(llevar_paraguas)
    red.agregar_nodo(utilidad)
    
    # Definir tablas de probabilidad.
    red.agregar_tabla_probabilidad(clima, {'soleado': 0.7, 'lluvioso': 0.3})
    red.agregar_tabla_probabilidad(pronostico, {
        ('soleado',): {'bueno': 0.8, 'malo': 0.2},
        ('lluvioso',): {'bueno': 0.3, 'malo': 0.7}
    })
    
    # Definir tabla de utilidad.
    red.agregar_tabla_utilidad(utilidad, {
        ('soleado', 'si'): -20,    # Llevar paraguas innecesariamente.
        ('soleado', 'no'): 100,   # No llevar y está soleado.
        ('lluvioso', 'si'): 80,   # Llevar paraguas y llueve.
        ('lluvioso', 'no'): -100  # No llevar y te mojas.
    })
    
    return red

if __name__ == "__main__":
    # Crear y resolver la red del ejemplo del paraguas.
    red_paraguas = ejemplo_paraguas()
    
    # Caso 1: Sin evidencia adicional.
    decision, utilidad = red_paraguas.mejor_decision()
    print("\nCaso 1: Sin evidencia adicional")
    print(f"Mejor decisión: {decision}")
    print(f"Utilidad esperada: {utilidad:.2f}")
    
    # Caso 2: Con pronóstico malo.
    decision, utilidad = red_paraguas.mejor_decision({'Pronostico': 'malo'})
    print("\nCaso 2: Con pronóstico malo")
    print(f"Mejor decisión: {decision}")
    print(f"Utilidad esperada: {utilidad:.2f}")
    
    # Caso 3: Con pronóstico bueno.
    decision, utilidad = red_paraguas.mejor_decision({'Pronostico': 'bueno'})
    print("\nCaso 3: Con pronóstico bueno")
    print(f"Mejor decisión: {decision}")
    print(f"Utilidad esperada: {utilidad:.2f}")