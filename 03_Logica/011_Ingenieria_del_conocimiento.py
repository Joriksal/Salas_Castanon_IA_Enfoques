import pandas as pd
from typing import Dict, List, Tuple

class IngenieriaConocimiento:
    """
    Clase para implementar un sistema de ingeniería del conocimiento.
    Permite extraer reglas basadas en datos y diagnosticar fallos a partir de síntomas observados.
    """
    def __init__(self, dataset: pd.DataFrame):
        """
        Inicializa el sistema con un conjunto de datos.
        
        Args:
            dataset (pd.DataFrame): DataFrame con columnas que representan síntomas y fallos.
        """
        self.dataset = dataset
        self.reglas = []  # Lista para almacenar las reglas extraídas
    
    def extraer_reglas(self, umbral_confianza: float = 0.6) -> List[str]:
        """
        Extrae reglas del tipo 'Si X entonces Y' con una confianza mayor o igual al umbral especificado.
        
        Args:
            umbral_confianza (float): Valor mínimo de confianza para considerar una regla válida.
        
        Returns:
            List[str]: Lista de reglas en formato legible.
        """
        for sintoma in self.dataset.columns:
            # Itera sobre las columnas que representan fallos
            for fallo in [col for col in self.dataset.columns if col.startswith("Fallo_")]:
                if sintoma != fallo:  # Evita comparar una columna consigo misma
                    confianza = self.calcular_confianza(sintoma, fallo)
                    if confianza >= umbral_confianza:
                        # Agrega la regla si la confianza cumple el umbral
                        self.reglas.append((sintoma, fallo, confianza))
        return self._formatear_reglas()
    
    def calcular_confianza(self, sintoma: str, fallo: str) -> float:
        """
        Calcula la confianza de la regla 'Si sintoma entonces fallo'.
        
        Args:
            sintoma (str): Nombre de la columna que representa el síntoma.
            fallo (str): Nombre de la columna que representa el fallo.
        
        Returns:
            float: Confianza de la regla, calculada como P(fallo | sintoma).
        """
        # Cuenta cuántas veces ocurre el síntoma
        total_sintoma = len(self.dataset[self.dataset[sintoma] == "Sí"])
        if total_sintoma == 0:
            return 0.0  # Evita división por cero si el síntoma no ocurre
        # Cuenta cuántas veces ocurre el fallo dado que el síntoma ocurre
        casos_fallo = len(self.dataset[
            (self.dataset[sintoma] == "Sí") & (self.dataset[fallo] == "Sí")
        ])
        return casos_fallo / total_sintoma  # Calcula la confianza
    
    def _formatear_reglas(self) -> List[str]:
        """
        Convierte las reglas almacenadas en un formato legible.
        
        Returns:
            List[str]: Lista de reglas formateadas como cadenas de texto.
        """
        return [
            f"Si {sintoma} entonces {fallo} (Confianza: {confianza:.0%})"
            for sintoma, fallo, confianza in self.reglas
        ]
    
    def diagnosticar(self, sintomas_observados: List[str]) -> Dict[str, Dict]:
        """
        Realiza un diagnóstico basado en los síntomas observados y las reglas extraídas.
        
        Args:
            sintomas_observados (List[str]): Lista de síntomas observados.
        
        Returns:
            Dict[str, Dict]: Diccionario con los fallos diagnosticados, su confianza y explicaciones.
        """
        resultados = {}
        for sintoma, fallo, confianza in self.reglas:
            if sintoma in sintomas_observados:  # Verifica si el síntoma está en los observados
                if fallo not in resultados:
                    # Si el fallo no está en los resultados, lo agrega con su confianza y explicación
                    resultados[fallo] = {
                        "confianza": confianza,
                        "explicacion": [f"Se observó {sintoma} (Confianza: {confianza:.0%})"]
                    }
                else:
                    # Si el fallo ya está en los resultados, actualiza la confianza máxima y agrega la explicación
                    resultados[fallo]["confianza"] = max(resultados[fallo]["confianza"], confianza)
                    resultados[fallo]["explicacion"].append(f"Se observó {sintoma} (Confianza: {confianza:.0%})")
        return resultados

# --- Uso del sistema ---
if __name__ == "__main__":
    # Dataset de ejemplo con síntomas y fallos
    data = {
        "Motor_Caliente": ["Sí", "Sí", "No", "Sí", "No"],
        "Fuga_Aceite": ["No", "Sí", "No", "No", "Sí"],
        "Vibración_Alta": ["Sí", "No", "Sí", "Sí", "No"],
        "Fallo_Motor": ["Sí", "Sí", "No", "Sí", "No"],
        "Fallo_Transmisión": ["No", "No", "Sí", "No", "Sí"]
    }
    df = pd.DataFrame(data)
    
    # Paso 1: Crear el sistema y extraer reglas
    sistema = IngenieriaConocimiento(df)
    reglas = sistema.extraer_reglas(umbral_confianza=0.5)
    print("=== Reglas extraídas ===")
    for regla in reglas:
        print(regla)
    
    # Paso 2: Diagnosticar con nuevos síntomas observados
    sintomas = ["Motor_Caliente", "Vibración_Alta"]
    diagnostico = sistema.diagnosticar(sintomas)
    
    print("\n=== Diagnóstico ===")
    for fallo, info in diagnostico.items():
        print(f"\n* {fallo}:")
        print(f"  - Confianza: {info['confianza']:.0%}")
        print("  - Explicación:")
        for linea in info["explicacion"]:
            print(f"    - {linea}")