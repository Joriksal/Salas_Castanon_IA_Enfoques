from typing import Dict, List, Set, Tuple, Optional  # Para definir tipos de datos como diccionarios, listas, conjuntos, tuplas y valores opcionales, mejorando la legibilidad y el control de tipos.

class SistemaNoMonotonico:
    def __init__(self):
        """
        Inicializa el sistema con las estructuras de datos necesarias:
        - `hechos`: Conjunto de hechos incontrovertibles (observaciones directas).
        - `reglas_default`: Lista de reglas por defecto en forma de tuplas (premisa, conclusión, justificación).
        - `creencias_justificadas`: Diccionario que almacena las creencias justificadas y sus justificaciones.
        - `contradicciones`: Conjunto de pares de creencias que se contradicen mutuamente.
        """
        self.hechos: Set[str] = set()  # Conjunto de hechos observados directamente
        self.reglas_default: List[Tuple[str, str, str]] = []  # Lista de reglas default
        self.creencias_justificadas: Dict[str, List[str]] = {}  # Creencias derivadas con sus justificaciones
        self.contradicciones: Set[Tuple[str, str]] = set()  # Pares de creencias mutuamente excluyentes

    def agregar_hecho(self, hecho: str):
        """
        Añade un hecho incontrovertible al sistema.
        - Los hechos son observaciones directas que no necesitan justificación.
        - Después de agregar un hecho, se revisan las creencias para asegurarse de que sigan siendo válidas.
        """
        self.hechos.add(hecho)  # Añadimos el hecho al conjunto de hechos
        self._revisar_creencias()  # Revisamos las creencias para eliminar las que sean inválidas

    def agregar_default(self, premisa: str, conclusion: str, justificacion: str):
        """
        Añade una regla default al sistema.
        - Una regla default tiene una premisa, una conclusión y una justificación.
        - Después de agregar la regla, se actualizan las creencias basándose en las reglas existentes.
        """
        self.reglas_default.append((premisa, conclusion, justificacion))  # Añadimos la regla a la lista
        self._actualizar_creencias()  # Derivamos nuevas creencias basándonos en las reglas

    def agregar_contradiccion(self, creencia1: str, creencia2: str):
        """
        Registra dos creencias como mutuamente excluyentes.
        - Si una creencia es verdadera, la otra debe ser falsa.
        - Después de registrar la contradicción, se revisan las creencias para eliminar las que sean contradictorias.
        """
        self.contradicciones.add((creencia1, creencia2))  # Añadimos la contradicción en ambas direcciones
        self.contradicciones.add((creencia2, creencia1))
        self._revisar_creencias()  # Revisamos las creencias para eliminar inconsistencias

    def _actualizar_creencias(self):
        """
        Deriva nuevas creencias aplicando las reglas default.
        - Una regla default se aplica si su premisa está en los hechos y su conclusión no contradice ningún hecho o creencia existente.
        - Si no hay contradicción, la conclusión se añade como una creencia justificada.
        """
        for premisa, conclusion, justificacion in self.reglas_default:
            # Verificamos si la premisa está en los hechos y la conclusión no está ya justificada
            if premisa in self.hechos and conclusion not in self.creencias_justificadas:
                # Comprobamos si la conclusión contradice algún hecho o creencia existente
                contradice = any(
                    (conclusion, c) in self.contradicciones 
                    for c in self.hechos.union(self.creencias_justificadas.keys())
                )
                # Si no hay contradicción, añadimos la conclusión como creencia justificada
                if not contradice:
                    self.creencias_justificadas[conclusion] = [justificacion]

    def _revisar_creencias(self):
        """
        Revisa y elimina creencias que ya no están justificadas.
        - Una creencia se elimina si contradice un hecho nuevo o si su justificación ya no es válida.
        """
        for creencia in list(self.creencias_justificadas.keys()):  # Iteramos sobre una copia de las claves
            # Verificamos si la creencia contradice algún hecho nuevo
            contradice_hecho = any(
                (creencia, h) in self.contradicciones 
                for h in self.hechos
            )
            
            # Verificamos si la justificación de la creencia sigue siendo válida
            justificacion_valida = any(
                premisa in self.hechos
                for premisa, conc, justif in self.reglas_default
                if conc == creencia and justif in self.creencias_justificadas.get(creencia, [])
            )
            
            # Si contradice un hecho o su justificación ya no es válida, eliminamos la creencia
            if contradice_hecho or not justificacion_valida:
                del self.creencias_justificadas[creencia]

    def obtener_creencias(self) -> Set[str]:
        """
        Retorna el conjunto actual de creencias.
        - Las creencias incluyen los hechos incontrovertibles y las creencias justificadas.
        """
        return self.hechos.union(self.creencias_justificadas.keys())

    def explicar(self, creencia: str) -> Optional[List[str]]:
        """
        Provee las justificaciones para una creencia específica.
        - Si la creencia es un hecho, se indica que es incontrovertible.
        - Si la creencia es justificada, se retorna su lista de justificaciones.
        """
        if creencia in self.hechos:
            return ["Hecho incontrovertible"]  # Los hechos no necesitan justificación
        return self.creencias_justificadas.get(creencia, None)  # Retornamos las justificaciones si existen

# ------------------------------------------
# Ejemplo: Sistema de Diagnóstico Médico
# ------------------------------------------
if __name__ == "__main__":
    # Mensaje inicial
    print("=== Sistema de Razonamiento No Monotónico ===")
    sistema = SistemaNoMonotonico()  # Creamos una instancia del sistema
    
    # 1. Definir reglas default
    sistema.agregar_default(
        premisa="paciente_tose",
        conclusion="tiene_resfriado",
        justificacion="Default: La tos sugiere resfriado"
    )
    
    sistema.agregar_default(
        premisa="paciente_fiebre",
        conclusion="tiene_infeccion",
        justificacion="Default: La fiebre sugiere infección"
    )
    
    # 2. Definir contradicciones
    sistema.agregar_contradiccion("tiene_resfriado", "tiene_alergia")
    sistema.agregar_contradiccion("tiene_infeccion", "es_sano")
    
    # 3. Agregar hechos observados
    print("\nCaso 1: Paciente con tos")
    sistema.agregar_hecho("paciente_tose")
    print("Creencias:", sistema.obtener_creencias())
    print("Explicación 'tiene_resfriado':", sistema.explicar("tiene_resfriado"))
    
    # 4. Nueva evidencia contradice creencia previa
    print("\nCaso 2: Se descubre que es alergia")
    sistema.agregar_hecho("tiene_alergia")
    print("Creencias:", sistema.obtener_creencias())
    print("Explicación 'tiene_resfriado':", sistema.explicar("tiene_resfriado"))
    
    # 5. Caso complejo con múltiples defaults
    print("\nCaso 3: Paciente con fiebre y tos")
    sistema.agregar_hecho("paciente_fiebre")
    print("Creencias:", sistema.obtener_creencias())
    print("Explicación 'tiene_infeccion':", sistema.explicar("tiene_infeccion"))
    
    # 6. Nueva evidencia invalida una conclusión
    print("\nCaso 4: Examen muestra que está sano")
    sistema.agregar_hecho("es_sano")
    print("Creencias:", sistema.obtener_creencias())
    print("Explicación 'tiene_infeccion':", sistema.explicar("tiene_infeccion"))