import uuid

from neural_recommendation.applications.interfaces.schemas import MoviePublic, MovieSchema
from neural_recommendation.domain.models.movie import Movie
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository


class CreateMovieUseCase:
    def __init__(self, movie_repository: MovieRepository):
        self.movie_repository = movie_repository

    async def execute(self, movie_data: MovieSchema) -> MoviePublic:
        existing_movie = await self.movie_repository.get_by_title(movie_data.title)
        if existing_movie:
            raise ValueError(f"Movie with title '{movie_data.title}' already exists")

        movie = Movie(
            id=uuid.uuid4(), title=movie_data.title, genres=movie_data.genres, embedding=movie_data.embedding or []
        )

        created_movie = await self.movie_repository.create(movie)

        return MoviePublic(id=created_movie.id, title=created_movie.title, genres=created_movie.genres)
