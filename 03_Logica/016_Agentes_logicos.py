from typing import Dict, Optional

class Ambiente:
    """Simula un entorno con estados, percepciones y reglas físicas."""
    def __init__(self):
        # Estado inicial del ambiente: ubicación del agente y si hay suciedad en cada lugar
        self.estado = {
            'ubicacion': 'A',  # El agente comienza en la ubicación 'A'
            'sucio': {'A': True, 'B': False}  # 'A' está sucio, 'B' está limpio
        }

    def percibir(self) -> Dict:
        """Retorna percepciones disponibles para el agente."""
        # El agente percibe su ubicación actual y si está sucio
        return {
            'ubicacion_actual': self.estado['ubicacion'],
            'sucio': self.estado['sucio'][self.estado['ubicacion']]
        }

    def ejecutar_accion(self, accion: str) -> bool:
        """Actualiza el estado del ambiente según la acción del agente."""
        # Si el agente decide aspirar, limpia la ubicación actual
        if accion == 'aspirar':
            self.estado['sucio'][self.estado['ubicacion']] = False
            return True
        # Si el agente decide moverse a la izquierda, cambia de ubicación
        elif accion == 'mover_izquierda':
            if self.estado['ubicacion'] == 'B':  # Solo puede moverse si está en 'B'
                self.estado['ubicacion'] = 'A'
                return True
        # Si el agente decide moverse a la derecha, cambia de ubicación
        elif accion == 'mover_derecha':
            if self.estado['ubicacion'] == 'A':  # Solo puede moverse si está en 'A'
                self.estado['ubicacion'] = 'B'
                return True
        return False  # Si la acción no es válida, no se realiza ningún cambio

class AgenteLogico:
    """Agente basado en lógica que toma decisiones usando reglas simbólicas."""
    def __init__(self):
        # Base de conocimiento del agente: reglas y hechos
        self.base_conocimiento = {
            'reglas': [
                # Si el lugar está sucio, la acción es aspirar
                {'si': ['sucio'], 'entonces': 'aspirar'},
                # Si el lugar está limpio y está en 'A', la acción es moverse a la derecha
                {'si': ['no sucio', 'ubicacion_actual:A'], 'entonces': 'mover_derecha'},
                # Si el lugar está limpio y está en 'B', la acción es moverse a la izquierda
                {'si': ['no sucio', 'ubicacion_actual:B'], 'entonces': 'mover_izquierda'}
            ],
            'hechos': set()  # Conjunto de hechos conocidos por el agente
        }

    def actualizar_creencias(self, percepcion: Dict):
        """Actualiza la base de conocimiento con nuevas percepciones."""
        # Reinicia los hechos conocidos y los actualiza con base en la percepción
        self.base_conocimiento['hechos'] = set()
        if percepcion['sucio']:
            self.base_conocimiento['hechos'].add('sucio')  # Agrega el hecho de que está sucio
        else:
            self.base_conocimiento['hechos'].add('no sucio')  # Agrega el hecho de que no está sucio
        # Agrega la ubicación actual como un hecho
        self.base_conocimiento['hechos'].add(f"ubicacion_actual:{percepcion['ubicacion_actual']}")

    def tomar_decision(self) -> Optional[str]:
        """Evalúa las reglas para decidir la acción óptima."""
        # Recorre las reglas y verifica si las premisas se cumplen en los hechos
        for regla in self.base_conocimiento['reglas']:
            if all(premisa in self.base_conocimiento['hechos'] for premisa in regla['si']):
                return regla['entonces']  # Retorna la acción correspondiente
        return None  # Si no se cumple ninguna regla, no toma acción

# --- Simulación ---
def simular_agente(pasos: int = 5):
    """Simula el comportamiento del agente lógico en el ambiente."""
    ambiente = Ambiente()  # Crea el ambiente
    agente = AgenteLogico()  # Crea el agente lógico

    for paso in range(1, pasos + 1):
        print(f"\n--- Paso {paso} ---")
        percepcion = ambiente.percibir()  # El agente percibe el ambiente
        print(f"Percepción: {percepcion}")

        agente.actualizar_creencias(percepcion)  # Actualiza sus creencias con base en la percepción
        accion = agente.tomar_decision()  # Toma una decisión basada en las reglas
        print(f"Base de conocimiento: {agente.base_conocimiento['hechos']}")

        if accion:
            print(f"Acción elegida: {accion}")
            ambiente.ejecutar_accion(accion)  # Ejecuta la acción en el ambiente
        else:
            print("No se pudo determinar una acción.")  # Si no hay acción, lo indica

        print(f"Estado del ambiente: {ambiente.estado}")  # Muestra el estado actual del ambiente

if __name__ == "__main__":
    simular_agente(pasos=5)  # Ejecuta la simulación con 5 pasos