# Importamos las librerías necesarias
import numpy as np  # Librería para cálculos numéricos y manejo de matrices
import matplotlib.pyplot as plt  # Librería para graficar y mostrar imágenes
from mpl_toolkits.mplot3d import Axes3D  # Herramientas para gráficos en 3D (no se usa en este código, pero podría ser útil)

# Clase para representar un vector en 3D
class Vector3:
    def __init__(self, x, y, z):
        # Constructor que inicializa las coordenadas del vector
        self.x = x
        self.y = y
        self.z = z
    
    def dot(self, other):
        # Producto punto entre dos vectores (multiplicación escalar)
        # Fórmula: x1*x2 + y1*y2 + z1*z2
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    def normalize(self):
        # Normalización del vector: convierte el vector en uno unitario (longitud = 1)
        # Fórmula: dividir cada componente por la longitud del vector
        length = np.sqrt(self.dot(self))  # Longitud del vector (raíz cuadrada del producto punto consigo mismo)
        return Vector3(self.x/length, self.y/length, self.z/length)

# Clase para representar una esfera en la escena
class Sphere:
    def __init__(self, center, radius, color, emission=0):
        # Constructor que inicializa las propiedades de la esfera
        self.center = center  # Centro de la esfera (Vector3)
        self.radius = radius  # Radio de la esfera
        self.color = color    # Color de la esfera (Vector3)
        self.emission = emission  # Intensidad de emisión de luz (si es una fuente de luz)

# Genera una dirección aleatoria uniforme dentro de una esfera unitaria
def random_in_unit_sphere():
    """
    Genera una dirección aleatoria uniforme dentro de una esfera unitaria.
    Esto se utiliza para simular rebotes de luz difusos.
    """
    while True:
        # Generar un punto aleatorio en el espacio 3D con coordenadas entre -1 y 1
        p = Vector3(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))
        if p.dot(p) < 1:  # Verifica que el punto esté dentro de la esfera (longitud < 1)
            return p.normalize()  # Normaliza el vector antes de devolverlo

# Función principal para trazar un rayo en la escena
def trace_ray(origin, direction, spheres, depth=0, max_depth=3):
    """
    Traza un rayo en la escena para calcular el color de un píxel.
    Utiliza el método Monte Carlo para simular rebotes de luz.
    
    Parámetros:
    - origin: Origen del rayo (Vector3)
    - direction: Dirección del rayo (Vector3)
    - spheres: Lista de esferas en la escena
    - depth: Profundidad actual del rayo (número de rebotes)
    - max_depth: Máxima profundidad permitida para los rebotes
    """
    if depth >= max_depth:  # Si se alcanza la profundidad máxima, devolver negro
        return Vector3(0,0,0)
    
    # Buscar la intersección más cercana entre el rayo y las esferas
    closest_t = np.inf  # Distancia más cercana inicializada como infinito
    hit_sphere = None  # Esfera que el rayo golpea más cerca
    for sphere in spheres:
        # Vector desde el origen del rayo al centro de la esfera
        oc = Vector3(origin.x-sphere.center.x, origin.y-sphere.center.y, origin.z-sphere.center.z)
        # Coeficientes de la ecuación cuadrática para intersección
        a = direction.dot(direction)  # a = dirección del rayo al cuadrado
        b = 2.0 * oc.dot(direction)  # b = 2 * (oc · dirección)
        c = oc.dot(oc) - sphere.radius*sphere.radius  # c = (oc · oc) - radio^2
        discriminant = b*b - 4*a*c  # Discriminante de la ecuación cuadrática
        if discriminant > 0:  # Si el discriminante es positivo, hay intersección
            t = (-b - np.sqrt(discriminant)) / (2*a)  # Solución más cercana
            if t < closest_t and t > 0.001:  # Ignorar intersecciones detrás del origen
                closest_t = t
                hit_sphere = sphere
    
    if hit_sphere is None:  # Si no hay intersección, devolver color del cielo
        return Vector3(0.2, 0.7, 0.8)  # Azul claro como color del cielo
    
    # Calcular el punto de intersección y la normal en ese punto
    hit_point = Vector3(
        origin.x + direction.x*closest_t,
        origin.y + direction.y*closest_t,
        origin.z + direction.z*closest_t
    )
    normal = Vector3(
        hit_point.x - hit_sphere.center.x,
        hit_point.y - hit_sphere.center.y,
        hit_point.z - hit_sphere.center.z
    ).normalize()
    
    # Si la esfera es una fuente de luz, devolver su emisión
    if hit_sphere.emission > 0:
        return Vector3(
            hit_sphere.color.x * hit_sphere.emission,
            hit_sphere.color.y * hit_sphere.emission,
            hit_sphere.color.z * hit_sphere.emission
        )
    
    # Rebote difuso: generar un nuevo rayo en una dirección aleatoria
    new_dir = random_in_unit_sphere()
    color = trace_ray(hit_point, new_dir, spheres, depth+1, max_depth)
    
    # Atenuar el color por el color de la esfera y el ángulo del rayo
    attenuation = hit_sphere.color
    cos_theta = new_dir.dot(normal)  # Ángulo entre el rayo y la normal
    return Vector3(
        color.x * attenuation.x * cos_theta,
        color.y * attenuation.y * cos_theta,
        color.z * attenuation.z * cos_theta
    )

# --- Escena de ejemplo ---
spheres = [
    Sphere(Vector3(0, -100.5, -1), 100, Vector3(0.8, 0.8, 0.8)),  # Piso
    Sphere(Vector3(0, 0, -1), 0.5, Vector3(0.8, 0.3, 0.3)),       # Esfera roja
    Sphere(Vector3(1, 0, -1), 0.5, Vector3(0.3, 0.8, 0.3)),       # Esfera verde
    Sphere(Vector3(-1, 0, -1), 0.5, Vector3(0.3, 0.3, 0.8)),      # Esfera azul
    Sphere(Vector3(0, 10, -1), 2, Vector3(1,1,1), emission=5)     # Fuente de luz
]

# --- Renderizado ---
width, height = 200, 100  # Resolución de la imagen
image = np.zeros((height, width, 3))  # Matriz para almacenar los colores de los píxeles
samples_per_pixel = 10  # Número de muestras por píxel (para suavizar el ruido)

# Iterar sobre cada píxel de la imagen
for y in range(height):
    for x in range(width):
        color = Vector3(0,0,0)  # Inicializar el color del píxel
        for _ in range(samples_per_pixel):  # Promediar múltiples muestras por píxel
            # Generar un rayo desde la cámara
            u = (x + np.random.random()) / width
            v = (y + np.random.random()) / height
            ray_dir = Vector3(u-0.5, v-0.5, -1).normalize()  # Dirección del rayo
            color_sample = trace_ray(Vector3(0,0,0), ray_dir, spheres)  # Traza el rayo
            color.x += color_sample.x
            color.y += color_sample.y
            color.z += color_sample.z
        # Promediar el color final del píxel
        image[y,x] = [color.x/samples_per_pixel, color.y/samples_per_pixel, color.z/samples_per_pixel]

# Mostrar la imagen renderizada
plt.imshow(np.clip(image, 0, 1))  # Limitar los valores de color entre 0 y 1
plt.axis('off')  # Ocultar los ejes
plt.show()