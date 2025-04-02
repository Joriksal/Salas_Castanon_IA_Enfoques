import copy
from collections import deque, defaultdict

class CutConditioningScheduler:
    def __init__(self, cursos, profesores, aulas, horarios):
        """
        Inicializa el planificador con:
        - cursos: Lista de cursos a programar.
        - profesores: Diccionario {curso: profesor}.
        - aulas: Lista de aulas disponibles.
        - horarios: Lista de franjas horarias.
        """
        self.cursos = cursos
        self.profesores = profesores
        self.aulas = aulas
        self.horarios = horarios
        
        # Variables: Cada curso necesita un aula y horario.
        self.variables = cursos
        
        # Dominios: Combinaciones posibles de aula × horario.
        self.dominios = {
            curso: [(aula, horario) for aula in aulas for horario in horarios]
            for curso in cursos
        }
        
        # Restricciones:
        self.restricciones = {}
        
        # 1. Un profesor no puede estar en dos lugares a la vez.
        for c1 in cursos:
            for c2 in cursos:
                if c1 != c2 and profesores[c1] == profesores[c2]:
                    self.restricciones[(c1, c2)] = lambda x, y: x[1] != y[1]
        
        # 2. Un aula no puede tener dos cursos simultáneamente.
        for c1 in cursos:
            for c2 in cursos:
                if c1 != c2:
                    self.restricciones[(c1, c2)] = lambda x, y: not (x[0] == y[0] and x[1] == y[1])
        
        # Vecinos para cada variable.
        self.vecinos = {curso: [] for curso in cursos}
        for (c1, c2) in self.restricciones:
            self.vecinos[c1].append(c2)
            self.vecinos[c2].append(c1)

    def consistente(self, var1, val1, var2, val2):
        """
        Verifica si dos asignaciones son consistentes con las restricciones.
        """
        if (var1, var2) in self.restricciones:
            if not self.restricciones[(var1, var2)](val1, val2):
                return False
        if (var2, var1) in self.restricciones:
            if not self.restricciones[(var2, var1)](val2, val1):
                return False
        return True

    def aplicar_acondicionamiento_corte(self):
        """
        Aplica el acondicionamiento del corte para reducir dominios.
        """
        # Primero aplicamos AC-3 para consistencia básica.
        if not self.ac3():
            return False
        
        # Luego aplicamos cortes específicos.
        cortes = self.identificar_cortes()
        for (curso, asignacion) in cortes:
            if asignacion in self.dominios[curso]:
                self.dominios[curso].remove(asignacion)
                if not self.dominios[curso]:
                    return False
        
        return True

    def ac3(self):
        """
        Algoritmo AC-3 para consistencia de arcos.
        """
        cola = deque(self.restricciones.keys())
        
        while cola:
            (xi, xj) = cola.popleft()
            if self.revisar_arco(xi, xj):
                if not self.dominios[xi]:
                    return False
                for xk in self.vecinos[xi]:
                    if xk != xj:
                        cola.append((xk, xi))
        return True

    def revisar_arco(self, xi, xj):
        """
        Elimina valores inconsistentes del dominio de xi respecto a xj.
        """
        revisado = False
        for x in list(self.dominios[xi]):
            if not any(self.consistente(xi, x, xj, y) for y in self.dominios[xj]):
                self.dominios[xi].remove(x)
                revisado = True
        return revisado

    def identificar_cortes(self):
        """
        Identifica asignaciones imposibles basadas en restricciones duras.
        """
        cortes = []
        
        # Cortes para profesores con disponibilidad limitada.
        disponibilidad_profesores = {
            prof: set(horario for curso, p in self.profesores.items() 
                     if p == prof for _, horario in self.dominios[curso])
            for prof in set(self.profesores.values())
        }
        
        for curso in self.cursos:
            profesor = self.profesores[curso]
            for (aula, horario) in list(self.dominios[curso]):
                # Si el profesor no tiene este horario disponible en otros cursos.
                if horario not in disponibilidad_profesores[profesor]:
                    cortes.append((curso, (aula, horario)))
        
        return cortes

    def resolver(self):
        """
        Resuelve el problema con backtracking + acondicionamiento.
        """
        if not self.aplicar_acondicionamiento_corte():
            return None
            
        return self.backtrack({})

    def backtrack(self, asignacion):
        """
        Backtracking con dominios reducidos.
        """
        if len(asignacion) == len(self.cursos):
            return asignacion
            
        curso = self.seleccionar_curso(asignacion)
        for (aula, horario) in self.ordenar_asignaciones(curso, asignacion):
            if self.consistente_con_asignacion(curso, (aula, horario), asignacion):
                asignacion[curso] = (aula, horario)
                
                # Guardar estado para backtracking.
                dominios_originales = copy.deepcopy(self.dominios)
                
                # Aplicar forward checking con protección.
                reducciones = self.forward_checking(asignacion, curso, (aula, horario))
                
                resultado = self.backtrack(asignacion)
                if resultado is not None:
                    return resultado
                    
                # Restaurar.
                del asignacion[curso]
                self.dominios = dominios_originales
                
        return None

    def forward_checking(self, asignacion, curso, asignacion_curso):
        """
        Elimina valores inconsistentes de cursos no asignados con verificación.
        """
        reducciones = defaultdict(list)
        aula, horario = asignacion_curso
        
        for vecino in self.vecinos[curso]:
            if vecino not in asignacion:
                for (a, h) in list(self.dominios[vecino]):
                    eliminar = False
                    
                    # Mismo profesor mismo horario.
                    if (self.profesores[curso] == self.profesores[vecino] and h == horario):
                        eliminar = True
                    
                    # Misma aula mismo horario.
                    if a == aula and h == horario:
                        eliminar = True
                    
                    if eliminar and (a, h) in self.dominios[vecino]:
                        reducciones[vecino].append((a, h))
                        self.dominios[vecino].remove((a, h))
        
        return reducciones

    def seleccionar_curso(self, asignacion):
        """
        Selecciona el curso con menos opciones disponibles (MRV).
        """
        no_asignados = [c for c in self.cursos if c not in asignacion]
        return min(no_asignados, key=lambda c: len(self.dominios[c]))

    def ordenar_asignaciones(self, curso, asignacion):
        """
        Ordena asignaciones por menor conflicto potencial.
        """
        return sorted(self.dominios[curso],
                    key=lambda x: self.num_conflictos(curso, x, asignacion))

    def num_conflictos(self, curso, asignacion_curso, asignacion):
        """
        Calcula conflictos potenciales.
        """
        aula, horario = asignacion_curso
        conflictos = 0
        
        for c in self.cursos:
            if c in asignacion:
                a, h = asignacion[c]
                # Mismo profesor mismo horario.
                if self.profesores[curso] == self.profesores[c] and h == horario:
                    conflictos += 1
                # Misma aula mismo horario.
                if a == aula and h == horario:
                    conflictos += 1
                    
        return conflictos

    def consistente_con_asignacion(self, curso, asignacion_curso, asignacion):
        """
        Verifica consistencia con asignación actual.
        """
        aula, horario = asignacion_curso
        
        for c, (a, h) in asignacion.items():
            # Mismo profesor mismo horario.
            if self.profesores[curso] == self.profesores[c] and h == horario:
                return False
            # Misma aula mismo horario.
            if a == aula and h == horario:
                return False
                
        return True

# --------------------------------------------
# Ejemplo de uso: Planificación de horarios
# --------------------------------------------

if __name__ == "__main__":
    # Datos de ejemplo modificados para asegurar solución.
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