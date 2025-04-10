import numpy as np
from collections import defaultdict

class MarkovBlanketRecommender:
    def __init__(self):
        # Estructura simplificada de la red bayesiana
        # Define las relaciones entre las variables en términos de mantos de Markov
        self.markov_blankets = {
            'Usuario': ['Edad', 'Rating', 'Película'],  # Variables relacionadas con el usuario
            'Película': ['Género', 'Rating'],          # Variables relacionadas con la película
            'Género': ['Película'],                    # Género depende de las películas
            'Rating': ['Usuario', 'Película'],         # Calificación depende del usuario y la película
            'Edad': ['Usuario']                        # Edad depende del usuario
        }
        
        # Tablas de probabilidad condicional (CPTs) simplificadas
        # Representan las probabilidades de cada variable en función de otras
        self.cpts = {
            'Género': {  # Probabilidad de cada género
                'Acción': 0.3,
                'Comedia': 0.4,
                'Drama': 0.3
            },
            'Película': {  # Probabilidad de películas dentro de cada género
                'Acción': {'Avengers': 0.4, 'JohnWick': 0.3, 'Inception': 0.3},
                'Comedia': {'Superbad': 0.5, 'Hangover': 0.3, 'Ted': 0.2},
                'Drama': {'Titanic': 0.6, 'Shawshank': 0.3, 'ForrestGump': 0.1}
            },
            'Usuario': {  # Probabilidad de usuarios según su grupo de edad
                'Joven': {'Alice': 0.4, 'Bob': 0.3, 'Charlie': 0.3},
                'Adulto': {'Alice': 0.2, 'Bob': 0.4, 'Charlie': 0.4},
                'Mayor': {'Alice': 0.3, 'Bob': 0.3, 'Charlie': 0.4}
            },
            'Rating': {  # Calificaciones dadas por cada usuario a las películas
                'Alice': {'Avengers': 0.7, 'Titanic': 0.8, 'Superbad': 0.9},
                'Bob': {'Avengers': 0.8, 'Hangover': 0.9, 'Shawshank': 0.7},
                'Charlie': {'Inception': 0.6, 'ForrestGump': 0.8, 'Ted': 0.5}
            },
            'Edad': {  # Grupo de edad de cada usuario
                'Alice': 'Joven',
                'Bob': 'Adulto',
                'Charlie': 'Mayor'
            }
        }
    
    def get_user_age(self, user):
        """Obtiene la edad del usuario a partir de las CPTs."""
        return self.cpts['Edad'].get(user, 'Adulto')  # Por defecto, 'Adulto'

    def get_movie_genre(self, movie):
        """Obtiene el género de una película específica."""
        for genre, movies in self.cpts['Película'].items():
            if movie in movies:
                return genre
        return 'Comedia'  # Por defecto, 'Comedia'

    def recommend_movie(self, user):
        """
        Recomienda películas basadas en el perfil del usuario.
        Utiliza una versión simplificada que no requiere evidencia completa.
        """
        # Obtener la edad del usuario
        age = self.get_user_age(user)
        
        # 1. Obtener películas que el usuario no ha calificado aún
        rated_movies = set(self.cpts['Rating'].get(user, {}).keys())  # Películas ya calificadas
        all_movies = set()  # Conjunto de todas las películas disponibles
        for genre_movies in self.cpts['Película'].values():
            all_movies.update(genre_movies.keys())
        unrated_movies = all_movies - rated_movies  # Películas no calificadas
        
        # 2. Predecir probabilidad de gustar cada película no calificada
        movie_scores = {}
        for movie in unrated_movies:
            genre = self.get_movie_genre(movie)  # Obtener género de la película
            
            # Peso basado en género preferido del usuario
            genre_weight = self.cpts['Género'].get(genre, 0.1)
            
            # Peso basado en calificaciones promedio del usuario
            avg_rating = np.mean(list(self.cpts['Rating'][user].values())) if user in self.cpts['Rating'] else 0.5
            movie_scores[movie] = genre_weight * avg_rating  # Calcular score para la película
        
        # 3. Ordenar películas por score en orden descendente
        ranked_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        return ranked_movies[:3]  # Retornar las 3 mejores recomendaciones

# Ejemplo de uso
if __name__ == "__main__":
    recommender = MarkovBlanketRecommender()
    
    # Recomendaciones para Alice
    print("Recomendaciones para Alice:")
    recommendations = recommender.recommend_movie('Alice')
    for movie, score in recommendations:
        print(f"- {movie}: {score:.4f}")
    
    # Recomendaciones para Bob
    print("\nRecomendaciones para Bob:")
    recommendations = recommender.recommend_movie('Bob')
    for movie, score in recommendations:
        print(f"- {movie}: {score:.4f}")
    
    # Recomendaciones para Charlie
    print("\nRecomendaciones para Charlie:")
    recommendations = recommender.recommend_movie('Charlie')
    for movie, score in recommendations:
        print(f"- {movie}: {score:.4f}")