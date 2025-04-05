import numpy as np
from collections import defaultdict

class RedBayesianaDinamica:
    def __init__(self, variables, arcos_temporales, arcos_intratemporales):
        """
        Inicializa una Red Bayesiana Dinámica (DBN).
        
        Args:
            variables: Lista de variables de la red.
            arcos_temporales: Lista de tuplas (padre_t, hijo_t+1) que conectan pasos consecutivos.
            arcos_intratemporales: Lista de tuplas (padre, hijo) dentro del mismo paso temporal.
        """
        self.variables = variables
        self.arcos_temporales = arcos_temporales
        self.arcos_intratemporales = arcos_intratemporales
        self.tablas_prob = defaultdict(dict)  # Tablas de probabilidad condicional (CPTs).
        self.pasos_temporales = 0  # Contador de pasos temporales.
        
        # Inicializar tablas de probabilidad vacías para cada variable.
        for var in variables:
            self.tablas_prob[var]['padres'] = []
            self.tablas_prob[var]['tabla'] = {}

    def agregar_tabla_probabilidad(self, variable, padres, tabla):
        """
        Agrega una tabla de probabilidad condicional (CPT) para una variable.
        
        Args:
            variable: Variable objetivo.
            padres: Lista de variables padres.
            tabla: Diccionario {tuple(valores_padres): distribucion}.
        """
        # Validar que las claves de la tabla sean tuplas.
        if not all(isinstance(k, tuple) for k in tabla.keys()):
            raise ValueError("Las claves de la tabla deben ser tuplas.")
        
        # Validar que los valores de la tabla sean distribuciones (diccionarios).
        if not all(isinstance(v, dict) for v in tabla.values()):
            raise ValueError("Los valores de la tabla deben ser distribuciones (diccionarios).")
        
        # Asignar la tabla de probabilidad a la variable.
        self.tablas_prob[variable]['padres'] = padres
        self.tablas_prob[variable]['tabla'] = tabla

    def transicion_estado(self, estado_actual):
        """
        Simula la transición al siguiente paso temporal.
        
        Args:
            estado_actual: Diccionario {variable: valor} que representa el estado actual.
            
        Returns:
            Dict: Nuevo estado en t+1.
        """
        nuevo_estado = {}

        # Procesar las variables que dependen del paso anterior (arcos temporales).
        for (padre, hijo) in self.arcos_temporales:
            if padre in estado_actual:
                # Obtener los valores de los padres (incluyendo padres intratemporales).
                padres_intra = [p for (p, h) in self.arcos_intratemporales if h == hijo]
                valores_padres = tuple([estado_actual[p] for p in [padre] + padres_intra])

                # Obtener la distribución condicional para el hijo.
                distribucion = self.tablas_prob[hijo]['tabla'].get(valores_padres, None)
                if distribucion:
                    # Muestrear un valor de la distribución condicional.
                    nuevo_estado[hijo] = self._muestrear_distribucion(distribucion)

        # Procesar las variables intratemporales (dentro del mismo paso).
        for var in self.variables:
            if var not in nuevo_estado:  # Si no se ha asignado un valor aún.
                padres = [p for (p, h) in self.arcos_intratemporales if h == var]
                if padres:
                    # Obtener los valores de los padres intratemporales.
                    valores_padres = tuple([estado_actual[p] for p in padres])
                    distribucion = self.tablas_prob[var]['tabla'].get(valores_padres, None)
                    if distribucion:
                        nuevo_estado[var] = self._muestrear_distribucion(distribucion)
                else:
                    # Si no tiene padres, usar la distribución marginal.
                    distribucion = self.tablas_prob[var]['tabla'].get((), None)
                    if distribucion:
                        nuevo_estado[var] = self._muestrear_distribucion(distribucion)

        self.pasos_temporales += 1  # Incrementar el contador de pasos temporales.
        return nuevo_estado

    def _muestrear_distribucion(self, distribucion):
        """
        Muestrea un valor de una distribución de probabilidad.
        
        Args:
            distribucion: Diccionario {valor: probabilidad}.
            
        Returns:
            Valor muestreado.
        """
        try:
            valores = list(distribucion.keys())
            probs = list(distribucion.values())
            
            # Normalizar probabilidades si no suman 1.
            suma = sum(probs)
            if suma <= 0:
                return np.random.choice(valores)
                
            probs_norm = [p / suma for p in probs]
            return np.random.choice(valores, p=probs_norm)
        except:
            return np.random.choice(valores)

    def inferencia_filtrado(self, observaciones, pasos):
        """
        Realiza inferencia por filtrado para estimar el estado actual dado observaciones.
        
        Args:
            observaciones: Lista de diccionarios {variable: valor} por paso temporal.
            pasos: Número de pasos a simular.
            
        Returns:
            List: Distribuciones de creencia por paso.
        """
        creencias = [{} for _ in range(pasos)]
        
        # Inicializar con distribuciones marginales de cada variable.
        for var in self.variables:
            # Obtener valores posibles de la tabla de probabilidad.
            if self.tablas_prob[var]['tabla']:
                # Tomar la primera distribución como referencia para los valores posibles.
                dist_ejemplo = next(iter(self.tablas_prob[var]['tabla'].values()))
                valores = list(dist_ejemplo.keys())
                prob_inicial = 1.0 / len(valores)
                
                for val in valores:
                    creencias[0][(var, val)] = prob_inicial
            else:
                # Valor por defecto si no hay tabla definida.
                creencias[0][(var, 'desconocido')] = 1.0
        
        for t in range(1, pasos):
            # Paso de predicción.
            pred = defaultdict(float)
            
            # Para cada variable en el paso actual.
            for var in self.variables:
                # Obtener padres temporales (del paso anterior).
                padres_temp = [p for (p, h) in self.arcos_temporales if h == var]
                
                if padres_temp:
                    # Variable con dependencia temporal.
                    for (var_ant, val_ant), prob_ant in creencias[t-1].items():
                        if var_ant in padres_temp:
                            key = (val_ant,)
                            distribucion = self.tablas_prob[var]['tabla'].get(key, None)
                            
                            if distribucion:
                                for val, prob_trans in distribucion.items():
                                    pred[(var, val)] += prob_ant * prob_trans
                            else:
                                # Distribución por defecto si no está definida.
                                valores = list(set(v for (v, _), p in creencias[t-1].items() if v == var))
                                if not valores:
                                    valores = ['default']
                                pred[(var, np.random.choice(valores))] += prob_ant
                else:
                    # Variable sin dependencia temporal (mantener distribución anterior).
                    for (v, val), prob in creencias[t-1].items():
                        if v == var:
                            pred[(var, val)] = prob
            
            # Paso de actualización con observación.
            obs = observaciones[t] if t < len(observaciones) else {}
            creencias[t] = defaultdict(float)
            
            # Normalizar predicción.
            total_pred = sum(pred.values())
            if total_pred > 0:
                for (var, val), prob in pred.items():
                    pred[(var, val)] = prob / total_pred
            
            # Aplicar observaciones.
            for (var, val), prob in pred.items():
                if var in obs:
                    # Modelo de observación con ruido.
                    if val == obs[var]:
                        creencias[t][(var, val)] = prob * 0.9  # Probabilidad alta si coincide.
                    else:
                        creencias[t][(var, val)] = prob * 0.1  # Probabilidad baja si no coincide.
                else:
                    creencias[t][(var, val)] = prob
            
            # Renormalizar.
            total = sum(creencias[t].values())
            if total > 0:
                for key in creencias[t]:
                    creencias[t][key] /= total
        
        return creencias

# --------------------------------------------
# Ejemplo: Sistema de Monitoreo de Salud
# --------------------------------------------
def ejemplo_monitoreo_salud():
    """
    Simula un sistema de monitoreo de salud usando una Red Bayesiana Dinámica.
    """
    # Variables del sistema.
    variables = ['Salud', 'Actividad', 'RitmoCardiaco', 'Observacion']

    # Arcos temporales (entre pasos).
    arcos_temporales = [('Salud', 'Salud'), ('Actividad', 'Actividad')]

    # Arcos intratemporales (dentro del mismo paso).
    arcos_intratemporales = [
        ('Salud', 'RitmoCardiaco'),
        ('Actividad', 'RitmoCardiaco'),
        ('RitmoCardiaco', 'Observacion')
    ]

    # Crear la Red Bayesiana Dinámica.
    dbn_salud = RedBayesianaDinamica(variables, arcos_temporales, arcos_intratemporales)

    # Definir tablas de probabilidad condicional.
    dbn_salud.agregar_tabla_probabilidad(
        'Salud',
        [('Salud', -1)],  # Padre del paso anterior.
        {
            ('buena',): {'buena': 0.8, 'mala': 0.2},
            ('mala',): {'buena': 0.3, 'mala': 0.7}
        }
    )

    dbn_salud.agregar_tabla_probabilidad(
        'Actividad',
        [('Actividad', -1)],
        {
            ('alta',): {'alta': 0.7, 'baja': 0.3},
            ('baja',): {'alta': 0.4, 'baja': 0.6}
        }
    )

    dbn_salud.agregar_tabla_probabilidad(
        'RitmoCardiaco',
        ['Salud', 'Actividad'],
        {
            ('buena', 'alta'): {'normal': 0.7, 'alto': 0.3},
            ('buena', 'baja'): {'normal': 0.9, 'alto': 0.1},
            ('mala', 'alta'): {'normal': 0.2, 'alto': 0.8},
            ('mala', 'baja'): {'normal': 0.5, 'alto': 0.5}
        }
    )

    dbn_salud.agregar_tabla_probabilidad(
        'Observacion',
        ['RitmoCardiaco'],
        {
            ('normal',): {'ok': 0.9, 'alerta': 0.1},
            ('alto',): {'ok': 0.2, 'alerta': 0.8}
        }
    )

    # Simular 5 pasos temporales.
    estado = {'Salud': 'buena', 'Actividad': 'alta', 'RitmoCardiaco': 'normal', 'Observacion': 'ok'}
    observaciones = []

    print("\nSimulación Temporal:")
    for t in range(5):
        print(f"\nPaso {t}:")
        print(f"Estado: {estado}")
        observaciones.append({'Observacion': estado['Observacion']})
        estado = dbn_salud.transicion_estado(estado)

    # Realizar inferencia por filtrado.
    print("\nInferencia por Filtrado:")
    creencias = dbn_salud.inferencia_filtrado(observaciones, pasos=5)
    for t, creencia in enumerate(creencias):
        print(f"\nPaso {t}:")
        # Agrupar por variable para mejor presentación.
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