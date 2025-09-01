import random
from typing import Dict, Optional

class Cocina:
    def __init__(self):
        self.ingredientes = {
            'pan': True,
            'jamon': True,
            'pasta': True,
            'salsa': False,
            'huevo': True
        }

    def percibir(self) -> Dict[str, bool]:
        return self.ingredientes.copy()

    def ejecutar_accion(self, accion: str):
        if accion == 'sandwich':
            self.ingredientes['pan'] = False
            self.ingredientes['jamon'] = False
        elif accion == 'pasta con salsa':
            self.ingredientes['pasta'] = False
            self.ingredientes['salsa'] = False
        elif accion == 'huevo frito':
            self.ingredientes['huevo'] = False

    def reponer_ingredientes(self, prob=0.4):
        """Simula ir al súper y reponer ingredientes con cierta probabilidad"""
        for ing in self.ingredientes:
            if not self.ingredientes[ing] and random.random() < prob:
                self.ingredientes[ing] = True
                print(f"   -> Se repuso: {ing}")


class AgentePlanificador:
    def __init__(self):
        self.base_conocimiento = {
            'recetas': [
                {'si': ['pan', 'jamon'], 'entonces': 'sandwich'},
                {'si': ['pasta', 'salsa'], 'entonces': 'pasta con salsa'},
                {'si': ['huevo'], 'entonces': 'huevo frito'}
            ],
            'hechos': set()
        }

    def actualizar_creencias(self, percepcion: Dict[str, bool]):
        self.base_conocimiento['hechos'] = {ing for ing, disponible in percepcion.items() if disponible}

    def tomar_decision(self) -> Optional[str]:
        for receta in self.base_conocimiento['recetas']:
            if all(ing in self.base_conocimiento['hechos'] for ing in receta['si']):
                return receta['entonces']
        return 'pedir comida'


def simular(pasos=6):
    cocina = Cocina()
    agente = AgentePlanificador()

    for paso in range(1, pasos + 1):
        print(f"\n--- Paso {paso} ---")
        percepcion = cocina.percibir()
        print(f"Ingredientes disponibles: {percepcion}")

        agente.actualizar_creencias(percepcion)
        print(f"Base de conocimiento: {agente.base_conocimiento['hechos']}")

        decision = agente.tomar_decision()
        print(f"Decisión: {decision}")

        if decision != 'pedir comida':
            cocina.ejecutar_accion(decision)

        cocina.reponer_ingredientes(prob=0.5)  # 50% de probabilidad de reponer


if __name__ == "__main__":
    simular()
