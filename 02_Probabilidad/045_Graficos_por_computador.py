import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Clase para representar un vector en 3D
class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
    
    # Producto punto entre dos vectores
    def dot(self, other):
        return self.x*other.x + self.y*other.y + self.z*other.z
    
    # Normalización del vector (convertirlo en un vector unitario)
    def normalize(self):
        length = np.sqrt(self.dot(self))
        return Vector3(self.x/length, self.y/length, self.z/length)

# Clase para representar una esfera en la escena
class Sphere:
    def __init__(self, center, radius, color, emission=0):
        self.center = center  # Centro de la esfera
        self.radius = radius  # Radio de la esfera
        self.color = color    # Color de la esfera
        self.emission = emission  # Intensidad de emisión de luz (si es una fuente de luz)

# Genera una dirección aleatoria uniforme dentro de una esfera unitaria
def random_in_unit_sphere():
    """Genera dirección aleatoria uniforme en esfera unitaria (para muestreo de luz)"""
    while True:
        p = Vector3(np.random.uniform(-1,1), np.random.uniform(-1,1), np.random.uniform(-1,1))
        if p.dot(p) < 1:  # Verifica que el punto esté dentro de la esfera
            return p.normalize()

# Función principal para trazar un rayo en la escena
def trace_ray(origin, direction, spheres, depth=0, max_depth=3):
    """Traza un rayo usando Monte Carlo"""
    if depth >= max_depth:  # Limita la profundidad de los rebotes
        return Vector3(0,0,0)  # Devuelve negro si se alcanza la profundidad máxima
    
    # Buscar la intersección más cercana con las esferas
    closest_t = np.inf
    hit_sphere = None
    for sphere in spheres:
        # Vector desde el origen del rayo al centro de la esfera
        oc = Vector3(origin.x-sphere.center.x, origin.y-sphere.center.y, origin.z-sphere.center.z)
        a = direction.dot(direction)
        b = 2.0 * oc.dot(direction)
        c = oc.dot(oc) - sphere.radius*sphere.radius
        discriminant = b*b - 4*a*c  # Fórmula cuadrática para intersección
        if discriminant > 0:  # Si hay intersección
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
    cos_theta = new_dir.dot(normal)
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