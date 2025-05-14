# Importamos la librería pandas para manejar y analizar datos estructurados en forma de DataFrame.
import pandas as pd

# Importamos tipos de datos de la librería typing para mejorar la legibilidad y robustez del código.
from typing import Dict, List, Tuple

class IngenieriaConocimiento:
    """
    Clase para implementar un sistema de ingeniería del conocimiento.
    Este sistema permite:
    1. Extraer reglas basadas en datos (relaciones entre síntomas y fallos).
    2. Diagnosticar fallos a partir de síntomas observados.
    """

    def __init__(self, dataset: pd.DataFrame):
        """
        Constructor de la clase. Inicializa el sistema con un conjunto de datos.

        Args:
            dataset (pd.DataFrame): DataFrame que contiene columnas que representan síntomas y fallos.
                                    Cada fila representa un caso o instancia.
        """
        self.dataset = dataset  # Almacena el DataFrame proporcionado.
        self.reglas = []  # Lista para almacenar las reglas extraídas del dataset.

    def extraer_reglas(self, umbral_confianza: float = 0.6) -> List[str]:
        """
        Extrae reglas del tipo 'Si X entonces Y' basadas en los datos del dataset.
        Solo se consideran reglas con una confianza mayor o igual al umbral especificado.

        Args:
            umbral_confianza (float): Valor mínimo de confianza para que una regla sea válida.
                                      Por defecto, es 0.6 (60%).

        Returns:
            List[str]: Lista de reglas en formato legible para el usuario.
        """
        # Iteramos sobre cada columna del dataset que representa un síntoma.
        for sintoma in self.dataset.columns:
            # Filtramos las columnas que representan fallos (nombres que comienzan con "Fallo_").
            for fallo in [col for col in self.dataset.columns if col.startswith("Fallo_")]:
                if sintoma != fallo:  # Evitamos comparar una columna consigo misma.
                    # Calculamos la confianza de la regla 'Si sintoma entonces fallo'.
                    confianza = self.calcular_confianza(sintoma, fallo)
                    if confianza >= umbral_confianza:
                        # Si la confianza cumple el umbral, agregamos la regla a la lista.
                        self.reglas.append((sintoma, fallo, confianza))
        # Formateamos las reglas en un formato legible antes de retornarlas.
        return self._formatear_reglas()

    def calcular_confianza(self, sintoma: str, fallo: str) -> float:
        """
        Calcula la confianza de la regla 'Si sintoma entonces fallo'.
        La confianza se define como la probabilidad condicional P(fallo | sintoma).

        Args:
            sintoma (str): Nombre de la columna que representa el síntoma.
            fallo (str): Nombre de la columna que representa el fallo.

        Returns:
            float: Confianza de la regla, calculada como P(fallo | sintoma).
                   Retorna 0.0 si el síntoma no ocurre en el dataset.
        """
        # Contamos cuántas veces ocurre el síntoma en el dataset.
        total_sintoma = len(self.dataset[self.dataset[sintoma] == "Sí"])
        if total_sintoma == 0:
            # Si el síntoma no ocurre, retornamos 0 para evitar división por cero.
            return 0.0
        # Contamos cuántas veces ocurre el fallo dado que el síntoma ocurre.
        casos_fallo = len(self.dataset[
            (self.dataset[sintoma] == "Sí") & (self.dataset[fallo] == "Sí")
        ])
        # Calculamos la confianza como la proporción de casos_fallo respecto a total_sintoma.
        return casos_fallo / total_sintoma

    def _formatear_reglas(self) -> List[str]:
        """
        Convierte las reglas almacenadas en un formato legible para el usuario.

        Returns:
            List[str]: Lista de reglas formateadas como cadenas de texto.
        """
        # Creamos una lista de cadenas con el formato 'Si X entonces Y (Confianza: Z%)'.
        return [
            f"Si {sintoma} entonces {fallo} (Confianza: {confianza:.0%})"
            for sintoma, fallo, confianza in self.reglas
        ]

    def diagnosticar(self, sintomas_observados: List[str]) -> Dict[str, Dict]:
        """
        Realiza un diagnóstico basado en los síntomas observados y las reglas extraídas.

        Args:
            sintomas_observados (List[str]): Lista de síntomas observados en el sistema.

        Returns:
            Dict[str, Dict]: Diccionario con los fallos diagnosticados.
                             Cada fallo incluye su confianza y una explicación basada en las reglas.
        """
        resultados = {}  # Diccionario para almacenar los resultados del diagnóstico.
        # Iteramos sobre las reglas extraídas.
        for sintoma, fallo, confianza in self.reglas:
            if sintoma in sintomas_observados:  # Verificamos si el síntoma está en los observados.
                if fallo not in resultados:
                    # Si el fallo no está en los resultados, lo agregamos con su confianza y explicación.
                    resultados[fallo] = {
                        "confianza": confianza,
                        "explicacion": [f"Se observó {sintoma} (Confianza: {confianza:.0%})"]
                    }
                else:
                    # Si el fallo ya está en los resultados, actualizamos la confianza máxima
                    # y agregamos la explicación correspondiente.
                    resultados[fallo]["confianza"] = max(resultados[fallo]["confianza"], confianza)
                    resultados[fallo]["explicacion"].append(f"Se observó {sintoma} (Confianza: {confianza:.0%})")
        return resultados


# --- Uso del sistema ---
if __name__ == "__main__":
    # Creamos un dataset de ejemplo con columnas que representan síntomas y fallos.
    data = {
        "Motor_Caliente": ["Sí", "Sí", "No", "Sí", "No"],  # Columna que indica si el motor está caliente.
        "Fuga_Aceite": ["No", "Sí", "No", "No", "Sí"],     # Columna que indica si hay fuga de aceite.
        "Vibración_Alta": ["Sí", "No", "Sí", "Sí", "No"],  # Columna que indica si hay vibración alta.
        "Fallo_Motor": ["Sí", "Sí", "No", "Sí", "No"],     # Columna que indica si hay fallo en el motor.
        "Fallo_Transmisión": ["No", "No", "Sí", "No", "Sí"]  # Columna que indica si hay fallo en la transmisión.
    }
    # Convertimos el diccionario en un DataFrame de pandas.
    df = pd.DataFrame(data)
    
    # Paso 1: Crear el sistema de ingeniería del conocimiento y extraer reglas.
    sistema = IngenieriaConocimiento(df)
    reglas = sistema.extraer_reglas(umbral_confianza=0.5)  # Umbral de confianza del 50%.
    print("=== Reglas extraídas ===")
    for regla in reglas:
        print(regla)  # Imprimimos cada regla extraída.
    
    # Paso 2: Diagnosticar con nuevos síntomas observados.
    sintomas = ["Motor_Caliente", "Vibración_Alta"]  # Lista de síntomas observados.
    diagnostico = sistema.diagnosticar(sintomas)  # Realizamos el diagnóstico.
    
    print("\n=== Diagnóstico ===")
    for fallo, info in diagnostico.items():
        print(f"\n* {fallo}:")  # Nombre del fallo diagnosticado.
        print(f"  - Confianza: {info['confianza']:.0%}")  # Confianza del diagnóstico.
        print("  - Explicación:")
        for linea in info["explicacion"]:
            print(f"    - {linea}")  # Explicación detallada del diagnóstico.