from pysat.solvers import Glucose3  # Importar el solucionador SAT Glucose3

# Definir variables proposicionales
# Cada variable representa un estado o acción en el problema
# Estado inicial y meta
start = 1  # El estado inicial del robot
goal = 2   # El estado objetivo que el robot debe alcanzar

# Acciones posibles en distintos tiempos
move_t1 = 3  # Acción: moverse en el tiempo 1
turn_t1 = 4  # Acción: girar en el tiempo 1
move_t2 = 5  # Acción: moverse en el tiempo 2

# Crear el solucionador SAT
solver = Glucose3()  # Instanciar el solucionador SAT

# Restricciones de estado inicial
# El robot comienza en el estado inicial
solver.add_clause([start])  # La cláusula asegura que el estado inicial es verdadero

# Restricciones de transición
# Estas cláusulas modelan cómo el robot puede cambiar de estado o realizar acciones
solver.add_clause([-start, move_t1])  # Si el robot está en el estado inicial, puede moverse en t1
solver.add_clause([-move_t1, move_t2])  # Si el robot se movió en t1, puede moverse en t2
solver.add_clause([-move_t2, goal])  # Si el robot se movió en t2, alcanza el estado objetivo

# Restricciones de exclusión mutua
# Estas cláusulas aseguran que ciertas acciones no puedan ocurrir simultáneamente
solver.add_clause([-move_t1, -turn_t1])  # El robot no puede moverse y girar al mismo tiempo en t1

# Buscar solución
# El solucionador intenta encontrar un modelo que satisfaga todas las cláusulas
if solver.solve():
    model = solver.get_model()  # Obtener el modelo que satisface las cláusulas
    # Filtrar las variables positivas, que representan las acciones tomadas en el plan
    plan = [var for var in model if var > 0]
    print("Plan encontrado:", plan)  # Imprimir el plan encontrado
else:
    print("No se encontró solución.")  # Indicar que no hay solución posible

# Liberar los recursos del solucionador
solver.delete()