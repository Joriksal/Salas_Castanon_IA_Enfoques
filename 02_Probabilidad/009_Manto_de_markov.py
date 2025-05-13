# Importamos la librería `numpy` para realizar operaciones matemáticas avanzadas, como el cálculo de promedios.
import numpy as np

# Definimos una clase llamada `MarkovBlanketRecommender` que encapsula toda la lógica del sistema de recomendación.
class MarkovBlanketRecommender:
    def __init__(self):
        """
        Constructor de la clase. Se ejecuta automáticamente al crear una instancia de la clase.
        Aquí inicializamos las estructuras de datos necesarias para el sistema de recomendación.
        """
        # Diccionario que define los mantos de Markov (Markov Blankets) para cada variable.
        # Un manto de Markov contiene todas las variables necesarias para determinar el estado de una variable específica.
        self.markov_blankets = {
            'Usuario': ['Edad', 'Rating', 'Película'],  # Variables relacionadas con el usuario.
            'Película': ['Género', 'Rating'],          # Variables relacionadas con la película.
            'Género': ['Película'],                    # El género depende de las películas.
            'Rating': ['Usuario', 'Película'],         # La calificación depende del usuario y la película.
            'Edad': ['Usuario']                        # La edad depende del usuario.
        }
        
        # Tablas de probabilidad condicional (CPTs, Conditional Probability Tables).
        # Estas tablas representan las probabilidades de cada variable en función de otras.
        self.cpts = {
            'Género': {  # Probabilidad de cada género.
                'Acción': 0.3,
                'Comedia': 0.4,
                'Drama': 0.3
            },
            'Película': {  # Probabilidad de películas dentro de cada género.
                'Acción': {'Avengers': 0.4, 'JohnWick': 0.3, 'Inception': 0.3},
                'Comedia': {'Superbad': 0.5, 'Hangover': 0.3, 'Ted': 0.2},
                'Drama': {'Titanic': 0.6, 'Shawshank': 0.3, 'ForrestGump': 0.1}
            },
            'Usuario': {  # Probabilidad de usuarios según su grupo de edad.
                'Joven': {'Alice': 0.4, 'Bob': 0.3, 'Charlie': 0.3},
                'Adulto': {'Alice': 0.2, 'Bob': 0.4, 'Charlie': 0.4},
                'Mayor': {'Alice': 0.3, 'Bob': 0.3, 'Charlie': 0.4}
            },
            'Rating': {  # Calificaciones dadas por cada usuario a las películas.
                'Alice': {'Avengers': 0.7, 'Titanic': 0.8, 'Superbad': 0.9},
                'Bob': {'Avengers': 0.8, 'Hangover': 0.9, 'Shawshank': 0.7},
                'Charlie': {'Inception': 0.6, 'ForrestGump': 0.8, 'Ted': 0.5}
            },
            'Edad': {  # Grupo de edad de cada usuario.
                'Alice': 'Joven',
                'Bob': 'Adulto',
                'Charlie': 'Mayor'
            }
        }
    
    def get_user_age(self, user):
        """
        Obtiene la edad del usuario a partir de las CPTs.
        Si el usuario no está en la tabla, se asume que es 'Adulto' por defecto.
        """
        return self.cpts['Edad'].get(user, 'Adulto')  # `.get` busca el valor asociado a la clave o devuelve un valor por defecto.

    def get_movie_genre(self, movie):
        """
        Obtiene el género de una película específica buscando en las CPTs.
        Si no se encuentra, se asume que el género es 'Comedia' por defecto.
        """
        for genre, movies in self.cpts['Película'].items():  # Iteramos sobre los géneros y sus películas.
            if movie in movies:  # Si la película está en la lista de un género, devolvemos ese género.
                return genre
        return 'Comedia'  # Género por defecto.

    def recommend_movie(self, user):
        """
        Recomienda películas basadas en el perfil del usuario.
        Utiliza una versión simplificada que no requiere evidencia completa.
        """
        # Obtener la edad del usuario.
        age = self.get_user_age(user)
        
        # 1. Obtener películas que el usuario no ha calificado aún.
        rated_movies = set(self.cpts['Rating'].get(user, {}).keys())  # Películas ya calificadas por el usuario.
        all_movies = set()  # Conjunto de todas las películas disponibles.
        for genre_movies in self.cpts['Película'].values():  # Iteramos sobre las películas de cada género.
            all_movies.update(genre_movies.keys())  # Añadimos las películas al conjunto.
        unrated_movies = all_movies - rated_movies  # Calculamos las películas no calificadas.
        
        # 2. Predecir probabilidad de gustar cada película no calificada.
        movie_scores = {}
        for movie in unrated_movies:  # Iteramos sobre las películas no calificadas.
            genre = self.get_movie_genre(movie)  # Obtener género de la película.
            
            # Peso basado en género preferido del usuario.
            genre_weight = self.cpts['Género'].get(genre, 0.1)  # Probabilidad del género.
            
            # Peso basado en calificaciones promedio del usuario.
            avg_rating = np.mean(list(self.cpts['Rating'][user].values())) if user in self.cpts['Rating'] else 0.5
            movie_scores[movie] = genre_weight * avg_rating  # Calcular score para la película.
        
        # 3. Ordenar películas por score en orden descendente.
        ranked_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)  # Ordenamos por score.
        
        return ranked_movies[:3]  # Retornar las 3 mejores recomendaciones.

# Ejemplo de uso del sistema de recomendación.
if __name__ == "__main__":  # Punto de entrada del programa. Este bloque se ejecuta solo si el archivo se ejecuta directamente.
    recommender = MarkovBlanketRecommender()  # Creamos una instancia de la clase.

    # Recomendaciones para Alice.
    print("Recomendaciones para Alice:")
    recommendations = recommender.recommend_movie('Alice')  # Llamamos al método de recomendación.
    for movie, score in recommendations:  # Iteramos sobre las recomendaciones.
        print(f"- {movie}: {score:.4f}")  # Mostramos la película y su score con 4 decimales.
    
    # Recomendaciones para Bob.
    print("\nRecomendaciones para Bob:")
    recommendations = recommender.recommend_movie('Bob')
    for movie, score in recommendations:
        print(f"- {movie}: {score:.4f}")
    
    # Recomendaciones para Charlie.
    print("\nRecomendaciones para Charlie:")
    recommendations = recommender.recommend_movie('Charlie')
    for movie, score in recommendations:
        print(f"- {movie}: {score:.4f}")