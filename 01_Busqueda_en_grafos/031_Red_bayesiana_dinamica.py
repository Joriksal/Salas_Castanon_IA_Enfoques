import numpy as np
# Importa numpy para operaciones numéricas.
from collections import defaultdict
# Importa defaultdict para diccionarios con valores por defecto.

class RedBayesianaDinamica:
    # Define la clase para la Red Bayesiana Dinámica.

    def __init__(self, variables, arcos_temporales, arcos_intratemporales):
        # Inicializa la red con variables, arcos temporales e intratemporales.
        self.variables = variables
        # Lista de nombres de las variables.
        self.arcos_temporales = arcos_temporales
        # Lista de tuplas (padre_t, hijo_t+1) que conectan pasos de tiempo.
        self.arcos_intratemporales = arcos_intratemporales
        # Lista de tuplas (padre, hijo) dentro del mismo paso de tiempo.
        self.tablas_prob = defaultdict(dict)
        # Diccionario para almacenar las tablas de probabilidad condicional (CPTs).
        self.pasos_temporales = 0
        # Contador de pasos de tiempo transcurridos.
        for var in variables:
            # Inicializa estructuras para cada variable.
            self.tablas_prob[var]['padres'] = []
            # Lista para los padres de la variable.
            self.tablas_prob[var]['tabla'] = {}
            # Diccionario para la tabla de probabilidad de la variable.

    def agregar_tabla_probabilidad(self, variable, padres, tabla):
        # Agrega una tabla de probabilidad condicional para una variable.
        if not all(isinstance(k, tuple) for k in tabla.keys()):
            raise ValueError("Las claves de la tabla deben ser tuplas.")
            # Valida que las claves de la tabla sean tuplas (combinaciones de valores de los padres).
        if not all(isinstance(v, dict) for v in tabla.values()):
            raise ValueError("Los valores de la tabla deben ser distribuciones (diccionarios).")
            # Valida que los valores de la tabla sean diccionarios (distribuciones de probabilidad).
        self.tablas_prob[variable]['padres'] = padres
        # Almacena la lista de padres para la variable.
        self.tablas_prob[variable]['tabla'] = tabla
        # Almacena la tabla de probabilidad para la variable.

    def transicion_estado(self, estado_actual):
        # Simula la transición al siguiente paso de tiempo.
        nuevo_estado = {}
        # Diccionario para el estado en el siguiente paso de tiempo.
        for padre, hijo in self.arcos_temporales:
            # Itera sobre los arcos que conectan pasos de tiempo.
            if padre in estado_actual:
                # Si el padre del arco temporal está en el estado actual.
                padres_intra = [p for (p, h) in self.arcos_intratemporales if h == hijo]
                # Encuentra padres intratemporales del hijo.
                valores_padres = tuple([estado_actual[p] for p in [padre] + padres_intra])
                # Obtiene los valores de los padres (temporal e intratemporales).
                distribucion = self.tablas_prob[hijo]['tabla'].get(valores_padres, None)
                # Obtiene la distribución de probabilidad condicional para el hijo.
                if distribucion:
                    nuevo_estado[hijo] = self._muestrear_distribucion(distribucion)
                    # Muestrea un valor para el hijo basado en la distribución.

        for var in self.variables:
            # Itera sobre todas las variables.
            if var not in nuevo_estado:
                # Si la variable aún no tiene valor en el nuevo estado.
                padres = [p for (p, h) in self.arcos_intratemporales if h == var]
                # Encuentra los padres intratemporales de la variable.
                if padres:
                    valores_padres = tuple([estado_actual[p] for p in padres])
                    # Obtiene los valores de los padres intratemporales.
                    distribucion = self.tablas_prob[var]['tabla'].get(valores_padres, None)
                    # Obtiene la distribución de probabilidad condicional.
                    if distribucion:
                        nuevo_estado[var] = self._muestrear_distribucion(distribucion)
                        # Muestrea un valor basado en la distribución.
                else:
                    distribucion = self.tablas_prob[var]['tabla'].get((), None)
                    # Obtiene la distribución marginal si no hay padres.
                    if distribucion:
                        nuevo_estado[var] = self._muestrear_distribucion(distribucion)
                        # Muestrea un valor basado en la distribución marginal.

        self.pasos_temporales += 1
        # Incrementa el contador de pasos de tiempo.
        return nuevo_estado
        # Retorna el nuevo estado del sistema.

    def _muestrear_distribucion(self, distribucion):
        # Muestrea un valor de una distribución de probabilidad.
        try:
            valores = list(distribucion.keys())
            # Obtiene la lista de posibles valores.
            probs = list(distribucion.values())
            # Obtiene la lista de probabilidades correspondientes.
            suma = sum(probs)
            # Calcula la suma de las probabilidades.
            if suma <= 0:
                return np.random.choice(valores)
                # Si la suma no es positiva, elige un valor aleatorio.
            probs_norm = [p / suma for p in probs]
            # Normaliza las probabilidades para que sumen 1.
            return np.random.choice(valores, p=probs_norm)
            # Elige un valor aleatorio basado en las probabilidades normalizadas.
        except:
            return np.random.choice(valores)
            # En caso de error, elige un valor aleatorio.

    def inferencia_filtrado(self, observaciones, pasos):
        # Realiza inferencia por filtrado para estimar el estado actual.
        creencias = [{} for _ in range(pasos)]
        # Lista para almacenar las distribuciones de creencia en cada paso.
        for var in self.variables:
            # Inicializa las creencias del primer paso con distribuciones marginales.
            if self.tablas_prob[var]['tabla']:
                dist_ejemplo = next(iter(self.tablas_prob[var]['tabla'].values()))
                valores = list(dist_ejemplo.keys())
                prob_inicial = 1.0 / len(valores)
                for val in valores:
                    creencias[0][(var, val)] = prob_inicial
            else:
                creencias[0][(var, 'desconocido')] = 1.0

        for t in range(1, pasos):
            # Itera sobre los pasos de tiempo para realizar la inferencia.
            pred = defaultdict(float)
            # Diccionario para almacenar las predicciones del estado actual.
            for var in self.variables:
                # Predicción basada en el paso anterior y arcos temporales.
                padres_temp = [p for (p, h) in self.arcos_temporales if h == var]
                if padres_temp:
                    for (var_ant, val_ant), prob_ant in creencias[t-1].items():
                        if var_ant in padres_temp:
                            key = (val_ant,)
                            distribucion = self.tablas_prob[var]['tabla'].get(key, None)
                            if distribucion:
                                for val, prob_trans in distribucion.items():
                                    pred[(var, val)] += prob_ant * prob_trans
                            else:
                                valores = list(set(v for (v, _), p in creencias[t-1].items() if v == var))
                                if not valores:
                                    valores = ['default']
                                pred[(var, np.random.choice(valores))] += prob_ant
                else:
                    # Si no hay dependencia temporal, se mantiene la distribución anterior.
                    for (v, val), prob in creencias[t-1].items():
                        if v == var:
                            pred[(var, val)] = prob

            # Actualización con la observación actual.
            obs = observaciones[t] if t < len(observaciones) else {}
            creencias[t] = defaultdict(float)
            total_pred = sum(pred.values())
            if total_pred > 0:
                for (var, val), prob in pred.items():
                    pred[(var, val)] = prob / total_pred

            for (var, val), prob in pred.items():
                if var in obs:
                    if val == obs[var]:
                        creencias[t][(var, val)] = prob * 0.9  # Mayor probabilidad si coincide.
                    else:
                        creencias[t][(var, val)] = prob * 0.1  # Menor probabilidad si no coincide.
                else:
                    creencias[t][(var, val)] = prob

            # Normalización de las creencias.
            total = sum(creencias[t].values())
            if total > 0:
                for key in creencias[t]:
                    creencias[t][key] /= total

        return creencias
        # Retorna la lista de distribuciones de creencia por paso.

def ejemplo_monitoreo_salud():
    # Ejemplo de uso: sistema de monitoreo de salud.
    variables = ['Salud', 'Actividad', 'RitmoCardiaco', 'Observacion']
    arcos_temporales = [('Salud', 'Salud'), ('Actividad', 'Actividad')]
    arcos_intratemporales = [
        ('Salud', 'RitmoCardiaco'),
        ('Actividad', 'RitmoCardiaco'),
        ('RitmoCardiaco', 'Observacion')
    ]
    dbn_salud = RedBayesianaDinamica(variables, arcos_temporales, arcos_intratemporales)
    dbn_salud.agregar_tabla_probabilidad(
        'Salud', [('Salud', -1)], {('buena',): {'buena': 0.8, 'mala': 0.2}, ('mala',): {'buena': 0.3, 'mala': 0.7}}
    )
    dbn_salud.agregar_tabla_probabilidad(
        'Actividad', [('Actividad', -1)], {('alta',): {'alta': 0.7, 'baja': 0.3}, ('baja',): {'alta': 0.4, 'baja': 0.6}}
    )
    dbn_salud.agregar_tabla_probabilidad(
        'RitmoCardiaco', ['Salud', 'Actividad'], {
            ('buena', 'alta'): {'normal': 0.7, 'alto': 0.3},
            ('buena', 'baja'): {'normal': 0.9, 'alto': 0.1},
            ('mala', 'alta'): {'normal': 0.2, 'alto': 0.8},
            ('mala', 'baja'): {'normal': 0.5, 'alto': 0.5}
        }
    )
    dbn_salud.agregar_tabla_probabilidad(
        'Observacion', ['RitmoCardiaco'], {('normal',): {'ok': 0.9, 'alerta': 0.1}, ('alto',): {'ok': 0.2, 'alerta': 0.8}}
    )

    estado = {'Salud': 'buena', 'Actividad': 'alta', 'RitmoCardiaco': 'normal', 'Observacion': 'ok'}
    observaciones = []
    print("\nSimulación Temporal:")
    for t in range(5):
        print(f"\nPaso {t}: Estado: {estado}")
        observaciones.append({'Observacion': estado['Observacion']})
        estado = dbn_salud.transicion_estado(estado)

    print("\nInferencia por Filtrado:")
    creencias = dbn_salud.inferencia_filtrado(observaciones, pasos=5)
    for t, creencia in enumerate(creencias):
        print(f"\nPaso {t}:")
        probs_por_variable = defaultdict(dict)
        for (var, val), prob in creencia.items():
            probs_por_variable[var][val] = prob
        for var in variables:
            print(f"\n{var}:")
            for val, prob in sorted(probs_por_variable.get(var, {}).items(), key=lambda x: -x[1]):
                print(f"  P({val}) = {prob:.2f}")

    return dbn_salud

if __name__ == "__main__":
    dbn = ejemplo_monitoreo_salud()