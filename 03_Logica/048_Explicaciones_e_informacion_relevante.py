# Clase que implementa un sistema explicativo basado en reglas.
class SistemaExplicativo:
    def __init__(self):
        # Inicializa el sistema con un conjunto de reglas predefinidas.
        # Cada regla contiene condiciones (síntomas) y un diagnóstico asociado.
        self.reglas = [
            {"condiciones": {"fiebre": "sí", "tos": "sí"}, "diagnóstico": "GRIPE"},
            {"condiciones": {"fiebre": "no", "dolor_cabeza": "no"}, "diagnóstico": "SANO"},
        ]

    # Método para diagnosticar con base en los síntomas proporcionados.
    def diagnosticar(self, sintomas):
        # Itera sobre las reglas para encontrar una que coincida con los síntomas.
        for regla in self.reglas:
            # Verifica si todos los síntomas coinciden con las condiciones de la regla.
            if all(sintomas.get(clave) == valor for clave, valor in regla["condiciones"].items()):
                # Genera una explicación basada en las condiciones de la regla.
                explicacion = self._explicar(regla["condiciones"])
                # Retorna el diagnóstico y la explicación.
                return regla["diagnóstico"], explicacion
        # Si no se encuentra una regla que coincida, retorna un diagnóstico desconocido.
        return "DESCONOCIDO", "No se encontró una regla que coincida con los síntomas."

    # Método privado para generar una explicación basada en las condiciones de una regla.
    def _explicar(self, condiciones):
        # Crea una lista de cadenas que describen las condiciones de la regla.
        partes = [f"{k} = {v}" for k, v in condiciones.items()]
        # Retorna una explicación en formato legible.
        return "Decisión basada en: " + ", ".join(partes)

# --- Ejemplo de uso ---
if __name__ == "__main__":
    # Crea una instancia del sistema explicativo.
    sistema = SistemaExplicativo()

    # Caso 1: Paciente con fiebre y tos.
    paciente_1 = {"fiebre": "sí", "tos": "sí", "dolor_cabeza": "no"}
    diag1, expl1 = sistema.diagnosticar(paciente_1)
    print(f"Paciente 1: Diagnóstico = {diag1}")
    print(f"Explicación: {expl1}\n")

    # Caso 2: Paciente sin fiebre, con tos, y sin dolor de cabeza.
    paciente_2 = {"fiebre": "no", "tos": "sí", "dolor_cabeza": "no"}
    diag2, expl2 = sistema.diagnosticar(paciente_2)
    print(f"Paciente 2: Diagnóstico = {diag2}")
    print(f"Explicación: {expl2}\n")

    # Caso 3: Paciente con fiebre, sin tos, y con dolor de cabeza.
    paciente_3 = {"fiebre": "sí", "tos": "no", "dolor_cabeza": "sí"}
    diag3, expl3 = sistema.diagnosticar(paciente_3)
    print(f"Paciente 3: Diagnóstico = {diag3}")
    print(f"Explicación: {expl3}")
