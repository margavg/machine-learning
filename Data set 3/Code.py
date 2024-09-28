import pandas as pd

# Cargar los datasets de películas y calificaciones
movies = pd.read_csv('movie.csv')
ratings = pd.read_csv('rating.csv')

# Filtrar películas que tengan el género 'Horror'
horror_movies = movies[movies['genres'].apply(lambda x: 'Horror' in x)]

# Unir películas de terror con las calificaciones
horror_ratings = ratings[ratings['movieId'].isin(horror_movies['movieId'])]

# Calcular promedio de calificación por película y ordenar
horror_avg_ratings = horror_ratings.groupby('movieId')['rating'].mean().reset_index()

# Añadir el título de la película
horror_avg_ratings = pd.merge(horror_avg_ratings, movies[['movieId', 'title']], on='movieId')

# Ordenar por promedio de calificación
horror_top_movies = horror_avg_ratings.sort_values(by='rating', ascending=False)

# Mostrar las mejores películas de terror
print(horror_top_movies.head(10))

# Obtener el ID de Toy Story
toy_story_id = movies[movies['title'].str.contains('Toy Story')].iloc[0]['movieId']

# Filtrar las calificaciones de Toy Story
toy_story_ratings = ratings[ratings['movieId'] == toy_story_id]

# Obtener los usuarios que han calificado Toy Story
users_who_rated_toy_story = toy_story_ratings['userId'].tolist()

# Calificaciones de otros usuarios que también vieron Toy Story
user_ratings = ratings[ratings['userId'].isin(users_who_rated_toy_story)]

# Calcular promedio de calificación y número de calificaciones por película
movie_recommendations = user_ratings.groupby('movieId').agg(
    avg_rating=('rating', 'mean'),
    count_ratings=('userId', 'size')
).reset_index()

# Unir con la tabla de películas
movie_recommendations = pd.merge(movie_recommendations, movies[['movieId', 'title']], on='movieId')

# Excluir Toy Story de las recomendaciones
movie_recommendations = movie_recommendations[movie_recommendations['movieId'] != toy_story_id]

# Ordenar por rating y número de calificaciones
movie_recommendations = movie_recommendations.sort_values(by=['avg_rating', 'count_ratings'], ascending=False)

# Mostrar las recomendaciones
print(movie_recommendations.head(10))