class SistemaLogico:
    """
    Sistema integrado que emula Prolog (Backward Chaining) y CLIPS (Forward Chaining).
    """

    def __init__(self):
        # Inicializa la base de conocimientos:
        # - `hechos`: Conjunto de hechos conocidos (como en CLIPS).
        # - `reglas`: Lista de reglas de inferencia (premisas -> conclusión).
        # - `objetos`: Diccionario para manejar templates y sus instancias (estilo CLIPS).
        self.hechos = set()  # Base de hechos (como en CLIPS)
        self.reglas = []     # Reglas de inferencia (premisas -> conclusión)
        self.objetos = {}    # Templates estilo CLIPS (opcional)

    # --- Métodos estilo CLIPS (Forward Chaining) ---
    def agregar_hecho(self, hecho: str):
        """
        Añade un hecho a la base de conocimientos.
        Ejemplo: "tiene_pelo(leon)"
        """
        self.hechos.add(hecho)

    def agregar_regla(self, premisas: list, conclusion: str):
        """
        Define una regla al estilo CLIPS: IF premisas THEN conclusión.
        Ejemplo: ["tiene_pelo(X)"] -> "es_mamifero(X)"
        """
        self.reglas.append((premisas, conclusion))

    def forward_chaining(self, verbose=False) -> set:
        """
        Implementa el encadenamiento hacia adelante (Forward Chaining).
        - Evalúa las reglas para derivar nuevos hechos a partir de los existentes.
        - Retorna todos los hechos derivados.
        """
        nuevos_hechos = True  # Bandera para seguir evaluando mientras haya nuevos hechos
        while nuevos_hechos:
            nuevos_hechos = False
            for premisas, conclusion in self.reglas:
                # Verifica si todas las premisas de una regla están en los hechos
                if all(p in self.hechos for p in premisas) and conclusion not in self.hechos:
                    # Si se cumplen las premisas, añade la conclusión como un nuevo hecho
                    self.hechos.add(conclusion)
                    nuevos_hechos = True
                    if verbose:
                        print(f"Regla aplicada: {premisas} -> {conclusion}")
                        print(f"Nuevo hecho: {conclusion}")
        return self.hechos

    # --- Métodos estilo Prolog (Backward Chaining) ---
    def backward_chaining(self, meta: str, visitados=None, verbose=False) -> bool:
        """
        Implementa el encadenamiento hacia atrás (Backward Chaining).
        - Verifica si una meta (objetivo) puede ser demostrada a partir de los hechos y reglas.
        """
        if visitados is None:
            visitados = set()  # Conjunto para evitar ciclos en la evaluación

        # Si la meta ya está en los hechos, se considera demostrada
        if meta in self.hechos:
            if verbose:
                print(f"Meta '{meta}' encontrada en hechos.")
            return True

        # Evita ciclos al verificar si la meta ya fue evaluada
        if meta in visitados:
            if verbose:
                print(f"Ciclo evitado: '{meta}' ya evaluada.")
            return False

        visitados.add(meta)  # Marca la meta como visitada

        # Busca reglas cuya conclusión sea la meta
        for premisas, conclusion in self.reglas:
            if conclusion == meta:
                if verbose:
                    print(f"\nEvaluando regla: {premisas} -> {conclusion}")
                premisas_cumplidas = True
                # Verifica recursivamente si todas las premisas se cumplen
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

        return False  # Si no se puede demostrar la meta, retorna False

    # --- Extensión: Templates estilo CLIPS ---
    def definir_template(self, nombre: str, slots: list):
        """
        Crea un template de objeto al estilo CLIPS.
        - `nombre`: Nombre del template.
        - `slots`: Lista de atributos que tendrán las instancias.
        """
        self.objetos[nombre] = {"slots": slots, "instancias": []}

    def agregar_instancia(self, template: str, valores: dict):
        """
        Añade una instancia de un template.
        - `template`: Nombre del template al que pertenece la instancia.
        - `valores`: Diccionario con los valores de los atributos (slots).
        """
        if template in self.objetos:
            self.objetos[template]["instancias"].append(valores)

# --- Ejemplo Integrado ---
if __name__ == "__main__":
    # Crea una instancia del sistema lógico
    sistema = SistemaLogico()

    print("==== SISTEMA EXPERTO INTEGRADO (Prolog + CLIPS) ====")
    print("\n1. Base de Conocimiento:")
    
    # Hechos iniciales (estilo CLIPS)
    sistema.agregar_hecho("tiene_pelo(leon)")
    sistema.agregar_hecho("tiene_plumas(condor)")
    sistema.agregar_hecho("pone_huevos(condor)")

    # Reglas (comunes para ambos enfoques)
    sistema.agregar_regla(["tiene_pelo(X)"], "es_mamifero(X)")
    sistema.agregar_regla(["tiene_plumas(X)"], "es_ave(X)")
    sistema.agregar_regla(["es_ave(X)", "vuela(X)"], "es_volador(X)")

    print("\n2. Forward Chaining (CLIPS):")
    # Encadenamiento hacia adelante para derivar nuevos hechos
    hechos_derivados = sistema.forward_chaining(verbose=True)
    print("\nHechos finales:", hechos_derivados)

    print("\n3. Backward Chaining (Prolog):")
    # Encadenamiento hacia atrás para demostrar una meta
    meta = "es_volador(condor)"
    resultado = sistema.backward_chaining(meta, verbose=True)
    print(f"\n¿Se puede demostrar '{meta}'? {'Sí' if resultado else 'No'}")

    print("\n4. Templates (CLIPS avanzado):")
    # Definición de un template y creación de instancias
    sistema.definir_template("animal", ["nombre", "tipo", "patas"])
    sistema.agregar_instancia("animal", {"nombre": "leon", "tipo": "mamifero", "patas": 4})
    print("Instancias de 'animal':", sistema.objetos["animal"]["instancias"])