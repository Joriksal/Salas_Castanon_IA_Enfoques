from collections import deque, defaultdict  # Importamos estructuras útiles para manejar colas y diccionarios con valores por defecto.

class CutConditioningScheduler:
    def __init__(self, cursos, profesores, aulas, horarios):
        """
        Inicializa el planificador con los datos necesarios:
        - cursos: Lista de cursos a programar.
        - profesores: Diccionario {curso: profesor}.
        - aulas: Lista de aulas disponibles.
        - horarios: Lista de franjas horarias.
        """
        self.cursos = cursos  # Lista de cursos.
        self.profesores = profesores  # Diccionario que asigna un profesor a cada curso.
        self.aulas = aulas  # Lista de aulas disponibles.
        self.horarios = horarios  # Lista de horarios disponibles.

        # Variables: Cada curso necesita un aula y un horario.
        self.variables = cursos

        # Dominios: Todas las combinaciones posibles de aula × horario para cada curso.
        self.dominios = {
            curso: [(aula, horario) for aula in aulas for horario in horarios]
            for curso in cursos
        }

        # Restricciones: Diccionario que define las restricciones entre pares de cursos.
        self.restricciones = {}

        # Restricción 1: Un profesor no puede estar en dos lugares al mismo tiempo.
        for c1 in cursos:
            for c2 in cursos:
                if c1 != c2 and profesores[c1] == profesores[c2]:
                    # La restricción asegura que los horarios de los cursos no coincidan.
                    self.restricciones[(c1, c2)] = lambda x, y: x[1] != y[1]

        # Restricción 2: Un aula no puede tener dos cursos simultáneamente.
        for c1 in cursos:
            for c2 in cursos:
                if c1 != c2:
                    # La restricción asegura que no se asignen el mismo aula y horario.
                    self.restricciones[(c1, c2)] = lambda x, y: not (x[0] == y[0] and x[1] == y[1])

        # Vecinos: Lista de cursos relacionados por restricciones.
        self.vecinos = {curso: [] for curso in cursos}
        for (c1, c2) in self.restricciones:
            self.vecinos[c1].append(c2)
            self.vecinos[c2].append(c1)

    def consistente(self, var1, val1, var2, val2):
        """
        Verifica si dos asignaciones son consistentes con las restricciones.
        - var1, var2: Cursos.
        - val1, val2: Asignaciones (aula, horario).
        """
        # Verifica si existe una restricción entre var1 y var2 y si se cumple.
        if (var1, var2) in self.restricciones:
            if not self.restricciones[(var1, var2)](val1, val2):
                return False
        # Verifica la restricción en el sentido inverso.
        if (var2, var1) in self.restricciones:
            if not self.restricciones[(var2, var1)](val2, val1):
                return False
        return True

    def aplicar_acondicionamiento_corte(self):
        """
        Aplica el acondicionamiento del corte para reducir los dominios de las variables.
        """
        # Primero aplica el algoritmo AC-3 para garantizar consistencia de arcos.
        if not self.ac3():
            return False

        # Luego identifica y elimina asignaciones imposibles (cortes).
        cortes = self.identificar_cortes()
        for (curso, asignacion) in cortes:
            if asignacion in self.dominios[curso]:
                self.dominios[curso].remove(asignacion)
                # Si un dominio queda vacío, no hay solución.
                if not self.dominios[curso]:
                    return False

        return True

    def ac3(self):
        """
        Algoritmo AC-3 para garantizar consistencia de arcos.
        """
        # Cola de arcos (pares de variables con restricciones).
        cola = deque(self.restricciones.keys())

        while cola:
            (xi, xj) = cola.popleft()  # Extrae un arco de la cola.
            if self.revisar_arco(xi, xj):
                # Si el dominio de xi queda vacío, no hay solución.
                if not self.dominios[xi]:
                    return False
                # Agrega los vecinos de xi a la cola para revisar consistencia.
                for xk in self.vecinos[xi]:
                    if xk != xj:
                        cola.append((xk, xi))
        return True

    def revisar_arco(self, xi, xj):
        """
        Revisa y elimina valores inconsistentes del dominio de xi respecto a xj.
        """
        revisado = False
        for x in list(self.dominios[xi]):
            # Si no hay ningún valor en xj que sea consistente con x, se elimina x.
            if not any(self.consistente(xi, x, xj, y) for y in self.dominios[xj]):
                self.dominios[xi].remove(x)
                revisado = True
        return revisado

    def identificar_cortes(self):
        """
        Identifica asignaciones imposibles basadas en restricciones duras.
        """
        cortes = []

        # Calcula la disponibilidad de horarios para cada profesor.
        disponibilidad_profesores = {
            prof: set(horario for curso, p in self.profesores.items()
                      if p == prof for _, horario in self.dominios[curso])
            for prof in set(self.profesores.values())
        }

        for curso in self.cursos:
            profesor = self.profesores[curso]
            for (aula, horario) in list(self.dominios[curso]):
                # Si el horario no está disponible para el profesor, se marca como corte.
                if horario not in disponibilidad_profesores[profesor]:
                    cortes.append((curso, (aula, horario)))

        return cortes

    def resolver(self):
        """
        Resuelve el problema utilizando backtracking con acondicionamiento del corte.
        """
        # Aplica el acondicionamiento del corte antes de iniciar el backtracking.
        if not self.aplicar_acondicionamiento_corte():
            return None

        # Inicia el proceso de backtracking.
        return self.backtrack({})

    def backtrack(self, asignacion):
        """
        Algoritmo de backtracking para encontrar una solución.
        - asignacion: Diccionario con las asignaciones actuales.
        """
        # Si todos los cursos están asignados, se retorna la solución.
        if len(asignacion) == len(self.cursos):
            return asignacion

        # Selecciona el siguiente curso a asignar usando la heurística MRV.
        curso = self.seleccionar_curso(asignacion)
        for (aula, horario) in self.ordenar_asignaciones(curso, asignacion):
            # Verifica si la asignación es consistente con las restricciones actuales.
            if self.consistente_con_asignacion(curso, (aula, horario), asignacion):
                asignacion[curso] = (aula, horario)

                # Guarda el estado actual de los dominios para restaurar si es necesario.
                dominios_originales = self.dominios.copy()

                # Aplica forward checking para reducir dominios.
                self.forward_checking(asignacion, curso, (aula, horario))

                # Llama recursivamente al backtracking.
                resultado = self.backtrack(asignacion)
                if resultado is not None:
                    return resultado

                # Si no se encuentra solución, deshace la asignación.
                del asignacion[curso]
                self.dominios = dominios_originales

        return None

    def forward_checking(self, asignacion, curso, asignacion_curso):
        """
        Aplica forward checking para eliminar valores inconsistentes de los dominios.
        """
        aula, horario = asignacion_curso

        for vecino in self.vecinos[curso]:
            if vecino not in asignacion:
                for (a, h) in list(self.dominios[vecino]):
                    # Elimina valores inconsistentes basados en restricciones.
                    if (self.profesores[curso] == self.profesores[vecino] and h == horario) or (a == aula and h == horario):
                        self.dominios[vecino].remove((a, h))

    def seleccionar_curso(self, asignacion):
        """
        Selecciona el curso no asignado con el menor número de opciones disponibles (heurística MRV).
        """
        no_asignados = [c for c in self.cursos if c not in asignacion]
        return min(no_asignados, key=lambda c: len(self.dominios[c]))

    def ordenar_asignaciones(self, curso, asignacion):
        """
        Ordena las asignaciones posibles para un curso según el menor conflicto potencial.
        """
        return sorted(self.dominios[curso], key=lambda x: self.num_conflictos(curso, x, asignacion))

    def num_conflictos(self, curso, asignacion_curso, asignacion):
        """
        Calcula el número de conflictos potenciales para una asignación.
        """
        aula, horario = asignacion_curso
        conflictos = 0

        for c in self.cursos:
            if c in asignacion:
                a, h = asignacion[c]
                # Incrementa conflictos si hay solapamiento de profesor o aula.
                if self.profesores[curso] == self.profesores[c] and h == horario:
                    conflictos += 1
                if a == aula and h == horario:
                    conflictos += 1

        return conflictos

    def consistente_con_asignacion(self, curso, asignacion_curso, asignacion):
        """
        Verifica si una asignación es consistente con las asignaciones actuales.
        """
        aula, horario = asignacion_curso

        for c, (a, h) in asignacion.items():
            # Verifica conflictos de profesor y aula.
            if self.profesores[curso] == self.profesores[c] and h == horario:
                return False
            if a == aula and h == horario:
                return False

        return True

# --------------------------------------------
# Ejemplo de uso: Planificación de horarios
# --------------------------------------------

if __name__ == "__main__":
    # Datos de ejemplo.
    cursos = ["Matemáticas", "Física", "Programación", "Historia"]
    profesores = {
        "Matemáticas": "Dr. Smith",
        "Física": "Dr. Johnson",
        "Programación": "Dr. Smith",
        "Historia": "Dra. Williams"
    }
    aulas = ["Aula 101", "Aula 202"]
    horarios = ["Lunes 9-11", "Lunes 11-13", "Martes 9-11"]

    print("Resolviendo problema de horarios universitarios con Acondicionamiento del Corte...")

    scheduler = CutConditioningScheduler(cursos, profesores, aulas, horarios)
    solucion = scheduler.resolver()

    if solucion:
        print("\nHorario asignado:")
        print("{:<15} {:<15} {:<15} {:<15}".format("Curso", "Profesor", "Aula", "Horario"))
        print("-" * 60)
        for curso in cursos:
            aula, horario = solucion[curso]
            print("{:<15} {:<15} {:<15} {:<15}".format(
                curso, profesores[curso], aula, horario))
    else:
        print("No se encontró una solución válida para los horarios.")