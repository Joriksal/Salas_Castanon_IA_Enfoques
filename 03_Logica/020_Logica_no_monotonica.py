from typing import Dict, List, Set, Tuple, Optional

class SistemaNoMonotonico:
    def __init__(self):
        # Conjunto de hechos incontrovertibles (observaciones directas)
        self.hechos: Set[str] = set()
        
        # Lista de reglas default en forma de tuplas (premisa, conclusión, justificación)
        self.reglas_default: List[Tuple[str, str, str]] = []
        
        # Diccionario que almacena las creencias justificadas y sus justificaciones
        self.creencias_justificadas: Dict[str, List[str]] = {}
        
        # Conjunto de pares de creencias que se contradicen mutuamente
        self.contradicciones: Set[Tuple[str, str]] = set()

    def agregar_hecho(self, hecho: str):
        """Añade un hecho incontrovertible al sistema."""
        self.hechos.add(hecho)
        # Revisa las creencias para asegurarse de que sigan siendo válidas
        self._revisar_creencias()

    def agregar_default(self, premisa: str, conclusion: str, justificacion: str):
        """Añade una regla default al sistema."""
        self.reglas_default.append((premisa, conclusion, justificacion))
        # Actualiza las creencias basándose en las nuevas reglas
        self._actualizar_creencias()

    def agregar_contradiccion(self, creencia1: str, creencia2: str):
        """Registra dos creencias como mutuamente excluyentes."""
        self.contradicciones.add((creencia1, creencia2))
        self.contradicciones.add((creencia2, creencia1))
        # Revisa las creencias para eliminar las que sean contradictorias
        self._revisar_creencias()

    def _actualizar_creencias(self):
        """Deriva nuevas creencias aplicando las reglas default."""
        for premisa, conclusion, justificacion in self.reglas_default:
            # Verifica si la premisa está en los hechos y la conclusión no está ya justificada
            if premisa in self.hechos and conclusion not in self.creencias_justificadas:
                # Comprueba si la conclusión contradice algún hecho o creencia existente
                contradice = any(
                    (conclusion, c) in self.contradicciones 
                    for c in self.hechos.union(self.creencias_justificadas.keys())
                )
                # Si no hay contradicción, añade la conclusión como creencia justificada
                if not contradice:
                    self.creencias_justificadas[conclusion] = [justificacion]

    def _revisar_creencias(self):
        """Revisa y elimina creencias que ya no están justificadas."""
        for creencia in list(self.creencias_justificadas.keys()):
            # Verifica si la creencia contradice algún hecho nuevo
            contradice_hecho = any(
                (creencia, h) in self.contradicciones 
                for h in self.hechos
            )
            
            # Verifica si la justificación de la creencia sigue siendo válida
            justificacion_valida = any(
                premisa in self.hechos
                for premisa, conc, justif in self.reglas_default
                if conc == creencia and justif in self.creencias_justificadas.get(creencia, [])
            )
            
            # Si contradice un hecho o su justificación ya no es válida, se elimina
            if contradice_hecho or not justificacion_valida:
                del self.creencias_justificadas[creencia]

    def obtener_creencias(self) -> Set[str]:
        """Retorna el conjunto actual de creencias (hechos + creencias justificadas)."""
        return self.hechos.union(self.creencias_justificadas.keys())

    def explicar(self, creencia: str) -> Optional[List[str]]:
        """Provee las justificaciones para una creencia específica."""
        if creencia in self.hechos:
            return ["Hecho incontrovertible"]  # Los hechos no necesitan justificación
        return self.creencias_justificadas.get(creencia, None)

# ------------------------------------------
# Ejemplo: Sistema de Diagnóstico Médico
# ------------------------------------------
if __name__ == "__main__":
    print("=== Sistema de Razonamiento No Monotónico ===")
    sistema = SistemaNoMonotonico()
    
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