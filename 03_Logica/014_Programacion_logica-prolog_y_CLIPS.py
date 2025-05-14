class SistemaLogico:
    """
    Clase que implementa un sistema lógico integrado que combina:
    - Encadenamiento hacia adelante (Forward Chaining) al estilo CLIPS.
    - Encadenamiento hacia atrás (Backward Chaining) al estilo Prolog.
    También incluye soporte para templates y objetos al estilo CLIPS.
    """

    def __init__(self):
        """
        Constructor de la clase SistemaLogico.
        Inicializa las estructuras de datos necesarias para manejar:
        - `hechos`: Base de hechos conocidos (estilo CLIPS).
        - `reglas`: Lista de reglas de inferencia (premisas -> conclusión).
        - `objetos`: Diccionario para manejar templates y sus instancias (estilo CLIPS).
        """
        self.hechos = set()  # Conjunto para almacenar hechos únicos.
        self.reglas = []     # Lista de reglas, cada una como un par (premisas, conclusión).
        self.objetos = {}    # Diccionario para manejar templates y sus instancias.

    # --- Métodos estilo CLIPS (Forward Chaining) ---
    def agregar_hecho(self, hecho: str):
        """
        Añade un hecho a la base de conocimientos.
        - `hecho`: Cadena que representa un hecho (ejemplo: "tiene_pelo(leon)").
        """
        self.hechos.add(hecho)  # Se usa un conjunto para evitar duplicados.

    def agregar_regla(self, premisas: list, conclusion: str):
        """
        Define una regla de inferencia al estilo CLIPS.
        - `premisas`: Lista de condiciones necesarias para aplicar la regla.
        - `conclusion`: Hecho que se deriva si se cumplen las premisas.
        Ejemplo: ["tiene_pelo(X)"] -> "es_mamifero(X)".
        """
        self.reglas.append((premisas, conclusion))  # Se almacena como un par (premisas, conclusión).

    def forward_chaining(self, verbose=False) -> set:
        """
        Implementa el encadenamiento hacia adelante (Forward Chaining).
        - Evalúa las reglas para derivar nuevos hechos a partir de los existentes.
        - Retorna todos los hechos derivados.
        - `verbose`: Si es True, imprime información detallada sobre las reglas aplicadas.
        """
        nuevos_hechos = True  # Bandera para controlar el ciclo de evaluación.
        while nuevos_hechos:  # Continúa mientras se deriven nuevos hechos.
            nuevos_hechos = False
            for premisas, conclusion in self.reglas:
                # Verifica si todas las premisas de una regla están en los hechos actuales.
                if all(p in self.hechos for p in premisas) and conclusion not in self.hechos:
                    # Si las premisas se cumplen, añade la conclusión como un nuevo hecho.
                    self.hechos.add(conclusion)
                    nuevos_hechos = True  # Indica que se derivaron nuevos hechos.
                    if verbose:
                        print(f"Regla aplicada: {premisas} -> {conclusion}")
                        print(f"Nuevo hecho: {conclusion}")
        return self.hechos  # Retorna la base de hechos actualizada.

    # --- Métodos estilo Prolog (Backward Chaining) ---
    def backward_chaining(self, meta: str, visitados=None, verbose=False) -> bool:
        """
        Implementa el encadenamiento hacia atrás (Backward Chaining).
        - Verifica si una meta (objetivo) puede ser demostrada a partir de los hechos y reglas.
        - `meta`: Meta que se desea demostrar (ejemplo: "es_volador(condor)").
        - `visitados`: Conjunto de metas ya evaluadas para evitar ciclos.
        - `verbose`: Si es True, imprime información detallada sobre el proceso de evaluación.
        """
        if visitados is None:
            visitados = set()  # Inicializa el conjunto de metas visitadas.

        # Si la meta ya está en los hechos, se considera demostrada.
        if meta in self.hechos:
            if verbose:
                print(f"Meta '{meta}' encontrada en hechos.")
            return True

        # Evita ciclos al verificar si la meta ya fue evaluada.
        if meta in visitados:
            if verbose:
                print(f"Ciclo evitado: '{meta}' ya evaluada.")
            return False

        visitados.add(meta)  # Marca la meta como visitada.

        # Busca reglas cuya conclusión sea la meta.
        for premisas, conclusion in self.reglas:
            if conclusion == meta:
                if verbose:
                    print(f"\nEvaluando regla: {premisas} -> {conclusion}")
                premisas_cumplidas = True
                # Verifica recursivamente si todas las premisas se cumplen.
                for p in premisas:
                    if verbose:
                        print(f"Verificando premisa: {p}")
                    if not self.backward_chaining(p, visitados, verbose):
                        premisas_cumplidas = False
                        break
                if premisas_cumplidas:
                    if verbose:
                        print(f"¡Meta '{meta}' demostrada!")
                    return True

        return False  # Si no se puede demostrar la meta, retorna False.

    # --- Extensión: Templates estilo CLIPS ---
    def definir_template(self, nombre: str, slots: list):
        """
        Crea un template de objeto al estilo CLIPS.
        - `nombre`: Nombre del template (ejemplo: "animal").
        - `slots`: Lista de atributos que tendrán las instancias (ejemplo: ["nombre", "tipo", "patas"]).
        """
        self.objetos[nombre] = {"slots": slots, "instancias": []}

    def agregar_instancia(self, template: str, valores: dict):
        """
        Añade una instancia de un template.
        - `template`: Nombre del template al que pertenece la instancia.
        - `valores`: Diccionario con los valores de los atributos (slots).
        Ejemplo: {"nombre": "leon", "tipo": "mamifero", "patas": 4}.
        """
        if template in self.objetos:
            self.objetos[template]["instancias"].append(valores)

# --- Ejemplo Integrado ---
if __name__ == "__main__":
    # Punto de entrada principal del programa.
    # Crea una instancia del sistema lógico.
    sistema = SistemaLogico()

    print("==== SISTEMA EXPERTO INTEGRADO (Prolog + CLIPS) ====")
    print("\n1. Base de Conocimiento:")
    
    # Hechos iniciales (estilo CLIPS).
    sistema.agregar_hecho("tiene_pelo(leon)")
    sistema.agregar_hecho("tiene_plumas(condor)")
    sistema.agregar_hecho("pone_huevos(condor)")

    # Reglas de inferencia (comunes para ambos enfoques).
    sistema.agregar_regla(["tiene_pelo(X)"], "es_mamifero(X)")
    sistema.agregar_regla(["tiene_plumas(X)"], "es_ave(X)")
    sistema.agregar_regla(["es_ave(X)", "vuela(X)"], "es_volador(X)")

    print("\n2. Forward Chaining (CLIPS):")
    # Encadenamiento hacia adelante para derivar nuevos hechos.
    hechos_derivados = sistema.forward_chaining(verbose=True)
    print("\nHechos finales:", hechos_derivados)

    print("\n3. Backward Chaining (Prolog):")
    # Encadenamiento hacia atrás para demostrar una meta.
    meta = "es_volador(condor)"
    resultado = sistema.backward_chaining(meta, verbose=True)
    print(f"\n¿Se puede demostrar '{meta}'? {'Sí' if resultado else 'No'}")

    print("\n4. Templates (CLIPS avanzado):")
    # Definición de un template y creación de instancias.
    sistema.definir_template("animal", ["nombre", "tipo", "patas"])
    sistema.agregar_instancia("animal", {"nombre": "leon", "tipo": "mamifero", "patas": 4})
    print("Instancias de 'animal':", sistema.objetos["animal"]["instancias"])