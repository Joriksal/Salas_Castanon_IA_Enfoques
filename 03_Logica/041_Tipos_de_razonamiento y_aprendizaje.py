import numpy as np
from sklearn.feature_extraction import DictVectorizer

class DoctorCabron:
    def __init__(self):
        # Inicialización de reglas basadas en conocimiento médico
        self.reglas = {
            ("fiebre", "tos", "dolor_garganta"): "GRIPE",
            ("fiebre", "erupcion_cutanea"): "VARICELA",
            ("dolor_cabeza", "nauseas", "sensibilidad_luz"): "MIGRAÑA"
        }
        # Tabla Q para aprendizaje por refuerzo
        self.Q = {}
        # Lista de posibles diagnósticos (acciones)
        self.acciones = list(set([d for d in self.reglas.values()] + ["DESCONOCIDO"]))
        # Vectorizador para convertir datos categóricos en vectores numéricos
        self.vectorizer = DictVectorizer(sparse=False)
        # Parámetros de aprendizaje por refuerzo
        self.alpha = 0.7  # Tasa de aprendizaje
        self.gamma = 0.9  # Factor de descuento
        # Historial de estados y acciones tomadas
        self.historial = []

    def diagnosticar(self, sintomas_paciente):
        """
        Diagnostica al paciente basado en reglas o aprendizaje por refuerzo.
        """
        # Diagnóstico basado en reglas
        diagnostico_reglas = next(
            (diag for sintomas, diag in self.reglas.items()
             if all(s in sintomas_paciente for s in sintomas)),
            "DESCONOCIDO"  # Si no hay coincidencia, retorna "DESCONOCIDO"
        )

        # Estado actual basado en los síntomas del paciente
        estado_actual = tuple(sorted(sintomas_paciente))

        # Inicializa la tabla Q para el estado actual si no existe
        if estado_actual not in self.Q:
            self.Q[estado_actual] = {a: 0 for a in self.acciones}

        # Selección de acción: exploración (20%) o explotación (80%)
        if np.random.random() < 0.2:
            diagnostico = np.random.choice(self.acciones)  # Exploración
        else:
            diagnostico = max(self.Q[estado_actual].items(), key=lambda x: x[1])[0]  # Explotación

        # Combina el diagnóstico basado en reglas con el de aprendizaje
        diagnostico_final = diagnostico_reglas if diagnostico_reglas != "DESCONOCIDO" else diagnostico

        # Guarda el estado y la acción en el historial
        self.historial.append({
            "estado": estado_actual,
            "accion": diagnostico_final
        })

        return diagnostico_final

    def actualizar_modelo(self, diagnostico_correcto, recompensa=10):
        """
        Actualiza el modelo Q-learning basado en el diagnóstico correcto y la recompensa.
        """
        if not self.historial:
            return  # Si no hay historial, no se puede actualizar

        # Recupera el último estado y acción del historial
        ultimo = self.historial[-1]
        estado = ultimo["estado"]
        accion = ultimo["accion"]

        # Inicializa la tabla Q para el estado si no existe
        if estado not in self.Q:
            self.Q[estado] = {a: 0 for a in self.acciones}

        # Ajusta la recompensa según si el diagnóstico fue correcto
        recompensa_ajustada = recompensa if accion == diagnostico_correcto else -5

        # Actualiza el valor Q usando la fórmula de Q-learning
        self.Q[estado][accion] += self.alpha * (
            recompensa_ajustada +
            self.gamma * max(self.Q[estado].values()) -  # Valor futuro esperado
            self.Q[estado][accion]  # Valor actual
        )

        # Aprende nuevas enfermedades si el diagnóstico correcto no está en las acciones
        if diagnostico_correcto not in self.acciones:
            self.acciones.append(diagnostico_correcto)
            for estado_Q in self.Q:
                self.Q[estado_Q][diagnostico_correcto] = 0

        # Agrega nuevas reglas basadas en el estado y el diagnóstico correcto
        if tuple(sorted(estado)) not in self.reglas.values():
            self.reglas[tuple(sorted(estado))] = diagnostico_correcto

if __name__ == "__main__":
    # Instancia del sistema de diagnóstico
    doc = DoctorCabron()

    # Caso 1: Diagnóstico basado en síntomas conocidos
    sintomas = ["fiebre", "tos", "dolor_garganta"]
    diag = doc.diagnosticar(sintomas)
    print(f"Paciente 1 ({sintomas}): Diagnóstico = {diag}")
    doc.actualizar_modelo("GRIPE")  # Actualiza el modelo con el diagnóstico correcto

    # Caso 2: Diagnóstico con síntomas nuevos y recompensa personalizada
    sintomas = ["mareo", "vision_borrosa"]
    diag = doc.diagnosticar(sintomas)
    print(f"Paciente 2 ({sintomas}): Diagnóstico = {diag}")
    doc.actualizar_modelo("DIABETES", recompensa=15)  # Nueva enfermedad aprendida

    # Caso 3: Diagnóstico tras aprendizaje
    sintomas = ["mareo", "vision_borrosa"]
    diag = doc.diagnosticar(sintomas)
    print(f"Paciente 3 ({sintomas}): Diagnóstico = {diag}")

    # Imprime las reglas aprendidas y los valores Q
    print("Reglas:", doc.reglas)
    print("Q-values:", {k: v for k, v in doc.Q.items() if any(vv != 0 for vv in v.values())})
