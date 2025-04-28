import time
from random import random
from collections import deque, defaultdict

class RobustExecutionMonitor:
    def __init__(self):
        # Definición de operadores con precondiciones, efectos, tasas de éxito y costos.
        # Estos operadores representan acciones que el robot puede realizar.
        self.operators = {
            # Operadores básicos de movimiento entre ubicaciones.
            "Mover_A_B": {
                "preconditions": ["RobotEnA"],  # Condiciones necesarias para ejecutar la acción.
                "effects": {"add": ["RobotEnB"], "remove": ["RobotEnA"]},  # Cambios en el estado tras la acción.
                "success_rate": 0.8,  # Probabilidad de éxito.
                "cost": 1  # Costo asociado a la acción.
            },
            "Mover_B_C": {
                "preconditions": ["RobotEnB"],
                "effects": {"add": ["RobotEnC"], "remove": ["RobotEnB"]},
                "success_rate": 0.85,
                "cost": 1
            },
            "Mover_C_A": {
                "preconditions": ["RobotEnC"],
                "effects": {"add": ["RobotEnA"], "remove": ["RobotEnC"]},
                "success_rate": 0.7,
                "cost": 1
            },
            
            # Operadores compuestos para transporte eficiente.
            "Transporte_Rapido_A_B": {
                "preconditions": ["RobotEnA", "PaqueteEnA"],
                "effects": {"add": ["RobotEnB", "PaqueteEnB"], "remove": ["RobotEnA", "PaqueteEnA"]},
                "success_rate": 0.7,
                "cost": 1.5
            },
            "Transporte_Seguro_A_B": {
                "preconditions": ["RobotEnA", "PaqueteEnA"],
                "effects": {"add": ["RobotEnB"], "remove": ["RobotEnA"]},
                "success_rate": 0.9,
                "cost": 2,
                "next_step": "Mover_Paquete_A_B"  # Acción encadenada tras el transporte seguro.
            },
            "Mover_Paquete_A_B": {
                "preconditions": ["RobotEnB", "PaqueteEnA"],
                "effects": {"add": ["PaqueteEnB"], "remove": ["PaqueteEnA"]},
                "success_rate": 0.95,
                "cost": 1
            },
            
            # Operadores de manipulación de paquetes.
            "Cargar_B": {
                "preconditions": ["RobotEnB", "PaqueteEnB"],
                "effects": {"add": ["PaqueteEnRobot"], "remove": ["PaqueteEnB"]},
                "success_rate": 0.95,
                "cost": 1
            },
            "Descargar_C": {
                "preconditions": ["RobotEnC", "PaqueteEnRobot"],
                "effects": {"add": ["PaqueteEnC"], "remove": ["PaqueteEnRobot"]},
                "success_rate": 0.98,
                "cost": 1
            }
        }

    def apply_operator(self, state, operator):
        """
        Aplica un operador al estado actual.
        Devuelve el nuevo estado, si tuvo éxito, y el operador ejecutado.
        """
        op = self.operators.get(operator, {})
        if not op:
            return set(state), False, operator  # Operador no encontrado.
            
        # Verificar si las precondiciones del operador se cumplen en el estado actual.
        if not all(precond in state for precond in op.get("preconditions", [])):
            return set(state), False, operator
            
        # Intentar ejecutar el operador con base en su tasa de éxito.
        success = random() <= op.get("success_rate", 0)
        if not success:
            return set(state), False, operator
            
        # Aplicar los efectos del operador al estado.
        new_state = set(state)
        new_state.update(op.get("effects", {}).get("add", []))
        new_state.difference_update(op.get("effects", {}).get("remove", []))
        
        # Si el operador tiene un paso siguiente encadenado, ejecutarlo.
        if "next_step" in op:
            return self.apply_operator(new_state, op["next_step"])
            
        return new_state, True, operator

    def find_alternative_actions(self, state, failed_op=None):
        """
        Encuentra acciones alternativas que se pueden ejecutar desde el estado actual.
        Las acciones se ordenan por costo.
        """
        alternatives = []
        for op_name, op in self.operators.items():
            if op_name == failed_op:  # Ignorar el operador que falló.
                continue
            if all(precond in state for precond in op.get("preconditions", [])):
                alternatives.append((op.get("cost", 1), op_name))
        return [op for _, op in sorted(alternatives)]  # Ordenar por costo.

    def robust_replan(self, goal, state, failed_op=None, depth=3, visited=None):
        """
        Realiza una replanificación robusta para alcanzar el objetivo desde el estado actual.
        Usa una búsqueda con profundidad limitada y evita ciclos.
        """
        if visited is None:
            visited = set()
            
        if depth <= 0:  # Limitar la profundidad de búsqueda.
            return None
            
        state_key = frozenset(state)
        if state_key in visited:  # Evitar ciclos.
            return None
        visited.add(state_key)
        
        # Verificar si el objetivo ya se cumplió.
        if all(g in state for g in goal):
            return []
            
        # Probar todas las acciones posibles ordenadas por costo.
        for action in self.find_alternative_actions(state, failed_op):
            new_state, success, _ = self.apply_operator(state, action)
            if not success:
                continue
                
            # Buscar un subplan recursivamente.
            subplan = self.robust_replan(goal, new_state, None, depth-1, visited)
            if subplan is not None:
                return [action] + subplan
                
        return None

    def execute_with_monitoring(self, goal, initial_plan, initial_state, max_attempts=5):
        """
        Ejecuta un plan con monitorización avanzada.
        Si una acción falla, intenta replanificar para alcanzar el objetivo.
        """
        history = defaultdict(int)  # Historial de ejecuciones de operadores.
        state = set(initial_state)  # Estado inicial.
        
        for attempt in range(1, max_attempts + 1):
            print(f"\n--- Intento {attempt} ---")
            current_plan = list(initial_plan)  # Copia del plan inicial.
            temp_state = set(state)  # Estado temporal para la ejecución.
            executed_ops = []  # Operadores ejecutados en este intento.
            
            while current_plan:
                op = current_plan.pop(0)  # Tomar la siguiente acción del plan.
                print(f"⚡ Ejecutando: {op}")
                time.sleep(0.3)  # Simular tiempo de ejecución.
                
                new_state, success, executed_op = self.apply_operator(temp_state, op)
                history[executed_op] += 1  # Registrar el operador ejecutado.
                
                if success:
                    temp_state = new_state  # Actualizar el estado.
                    executed_ops.append(executed_op)
                    print(f"  ✅ Éxito. Estado actual: {temp_state}")
                    
                    # Verificar si el objetivo se alcanzó.
                    if all(g in temp_state for g in goal):
                        print("\n🎯 Objetivo alcanzado!")
                        return True
                else:
                    print(f"  ❌ Fallo en {executed_op}. Replanificando...")
                    new_plan = self.robust_replan(goal, temp_state, executed_op)
                    
                    if new_plan:
                        print(f"  Nuevo plan: {new_plan}")
                        current_plan = new_plan + current_plan  # Insertar el nuevo plan.
                    else:
                        print("  No se encontró plan alternativo viable")
                        break
            
            # Si el objetivo se alcanzó, finalizar.
            if all(g in temp_state for g in goal):
                return True
                
            print("\n↻ Reiniciando ejecución...")
            # Penalizar operadores que fallan frecuentemente.
            for op in history:
                if history[op] > 1 and op in self.operators:
                    self.operators[op]["cost"] += 0.5  # Incrementar el costo.
        
        print("\n⚠️ Se agotaron todos los intentos")
        return False

# --- Ejemplo de uso completo ---
if __name__ == "__main__":
    print("=== SISTEMA DE PLANIFICACIÓN ROBUSTA ===")
    planner = RobustExecutionMonitor()
    
    # Definición de escenarios de prueba.
    escenarios = [
        {
            "nombre": "Caso 1: Transporte directo desde B",
            "estado": ["RobotEnA", "PaqueteEnB"],
            "objetivo": ["PaqueteEnC"],
            "plan": ["Mover_A_B", "Cargar_B", "Mover_B_C", "Descargar_C"]
        },
        {
            "nombre": "Caso 2: Transporte desde A con alternativas", 
            "estado": ["RobotEnA", "PaqueteEnA"],
            "objetivo": ["PaqueteEnC"],
            "plan": ["Transporte_Rapido_A_B", "Cargar_B", "Mover_B_C", "Descargar_C"]
        }
    ]

    # Ejecutar cada escenario.
    for escenario in escenarios:
        print(f"\n{'='*60}")
        print(f"{escenario['nombre']}")
        print(f"● Estado inicial: {escenario['estado']}")
        print(f"● Objetivo: {escenario['objetivo']}")
        print(f"● Plan inicial: {escenario['plan']}")
        print("-"*60)
        
        resultado = planner.execute_with_monitoring(
            escenario['objetivo'],
            escenario['plan'],
            escenario['estado']
        )
        
        print(f"\n● Resultado final: {'ÉXITO ✅' if resultado else 'FALLO ❌'}")
        print("="*60)