def parsear_regla(regla):
    """
    Divide una regla en dos partes: el antecedente (un conjunto de condiciones) 
    y el consecuente (una conclusión).
    
    Ejemplo:
    Entrada: "si A y B entonces C"
    Salida: ({"A", "B"}, "C")
    """
    # Dividir la regla en dos partes: antes y después de "entonces"
    partes = regla.split(" entonces ")
    # Extraer el antecedente eliminando "si " y separando por " y "
    antecedente = set(partes[0][3:].split(" y "))  # "si A y B" → {"A", "B"}
    # Extraer el consecuente eliminando espacios en blanco
    consecuente = partes[1].strip()                # "entonces C" → "C"
    return antecedente, consecuente

def diagnosticar(hechos_iniciales, reglas):
    """
    Realiza un diagnóstico aplicando reglas lógicas a un conjunto inicial de hechos.
    
    Parámetros:
    - hechos_iniciales: Un conjunto de hechos conocidos inicialmente.
    - reglas: Una lista de reglas en formato "si A y B entonces C".
    
    Retorna:
    - Un conjunto de hechos que incluye los iniciales y los derivados.
    """
    # Convertir los hechos iniciales en un conjunto para facilitar las operaciones
    hechos = set(hechos_iniciales)
    cambiado = True  # Variable para controlar si se han aplicado nuevas reglas
    
    # Ciclo para aplicar reglas hasta que no haya cambios
    while cambiado:
        cambiado = False  # Reiniciar el indicador de cambio
        for regla in reglas:
            # Parsear la regla en antecedente y consecuente
            antecedente, consecuente = parsear_regla(regla)
            # Verificar si el antecedente está contenido en los hechos actuales
            # y si el consecuente aún no está en los hechos
            if antecedente.issubset(hechos) and consecuente not in hechos:
                # Agregar el consecuente a los hechos
                hechos.add(consecuente)
                # Mostrar la regla aplicada y los hechos actualizados
                print(f"Regla aplicada: '{regla}'. Hechos actualizados: {hechos}")
                cambiado = True  # Indicar que hubo un cambio
    return hechos  # Retornar el conjunto final de hechos

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Lista de reglas en formato "si ... entonces ..."
    reglas = [
        "si fiebre y dolor_cabeza entonces gripe",
        "si fiebre y tos entonces resfriado",
        "si gripe entonces dolor_muscular"
    ]
    # Conjunto inicial de síntomas conocidos
    sintomas = {"fiebre", "dolor_cabeza"}
    
    # Aplicar el sistema de diagnóstico
    diagnostico = diagnosticar(sintomas, reglas)
    # Mostrar el diagnóstico final
    print(f"\nDiagnóstico final: {diagnostico}")