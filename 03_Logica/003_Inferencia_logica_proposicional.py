def forma_normal_conjuntiva(clausula):
    """
    Convierte una cláusula lógica a Forma Normal Conjuntiva (FNC).
    Ejemplo: "¬(p ∨ q)" → ["¬p", "¬q"].
    Nota: En este caso, asumimos que la entrada ya está en FNC como lista de literales.
    """
    return clausula

def resolver(clausula1, clausula2):
    """
    Aplica la regla de resolución a dos cláusulas.
    Busca pares complementarios (ejemplo: p y ¬p) y genera una nueva cláusula resolvente.
    Retorna la cláusula resultante o None si no hay resolución posible.
    """
    resolvente = None
    for literal in clausula1:
        # Caso 1: Encontrar un par complementario (ejemplo: p y ¬p)
        if f"¬{literal}" in clausula2:
            # Crear la cláusula resolvente eliminando el par complementario
            resolvente = [l for l in clausula1 if l != literal] + [l for l in clausula2 if l != f"¬{literal}"]
            break
        # Caso 2: Encontrar el complemento en el formato opuesto (ejemplo: ¬p y p)
        elif literal.startswith("¬") and literal[1:] in clausula2:
            resolvente = [l for l in clausula1 if l != literal] + [l for l in clausula2 if l != literal[1:]]
            break
    return resolvente

def inferencia_resolucion(base_conocimiento, consulta):
    """
    Realiza inferencia por resolución para determinar si la consulta se sigue de la base de conocimiento.
    Utiliza el método de resolución para buscar una contradicción.
    
    Parámetros:
    - base_conocimiento: Lista de cláusulas en FNC (Forma Normal Conjuntiva).
    - consulta: Literal que se desea verificar si es inferible.

    Retorna:
    - True si la consulta es válida (se puede inferir).
    - False si la consulta no es válida (no se puede inferir).
    """
    # Paso 1: Negar la consulta y añadirla a la base de conocimiento
    # Si la consulta es "p", añadimos "¬p". Si es "¬p", añadimos "p".
    clausulas = base_conocimiento.copy()
    clausulas.append([f"¬{consulta}"] if not consulta.startswith("¬") else [consulta[1:]])
    
    # Paso 2: Aplicar resolución iterativamente
    nuevas_clausulas = []
    while True:
        # Comparar todas las combinaciones de cláusulas
        for i in range(len(clausulas)):
            for j in range(i + 1, len(clausulas)):
                # Intentar resolver las cláusulas actuales
                resolvente = resolver(clausulas[i], clausulas[j])
                if resolvente == []:  # Contradicción encontrada (cláusula vacía)
                    return True  # La consulta es válida
                # Añadir la nueva cláusula si es válida y no está duplicada
                if resolvente and resolvente not in clausulas + nuevas_clausulas:
                    nuevas_clausulas.append(resolvente)
        
        # Si no se pueden generar más cláusulas, detener el proceso
        if not nuevas_clausulas:
            return False  # La consulta no es válida
        
        # Actualizar la base de conocimiento con las nuevas cláusulas generadas
        clausulas += nuevas_clausulas
        nuevas_clausulas = []

# Ejemplo de uso
if __name__ == "__main__":
    # Base de conocimiento en FNC (lista de cláusulas)
    # Ejemplo: (p ∨ q) ∧ (¬q ∨ r) ∧ (¬r)
    base_conocimiento = [
        ["p", "q"],  # Representa (p ∨ q)
        ["¬q", "r"], # Representa (¬q ∨ r)
        ["¬r"]       # Representa (¬r)
    ]
    
    # Consulta: ¿Se puede inferir "p" a partir de la base de conocimiento?
    consulta = "p"
    resultado = inferencia_resolucion(base_conocimiento, consulta)
    print(f"¿La consulta '{consulta}' es válida? {resultado}")  # True