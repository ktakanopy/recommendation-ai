import uuid

from neural_recommendation.applications.interfaces.schemas import MoviePublic, MovieSchema
from neural_recommendation.domain.models.movie import Movie
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository


class UpdateMovieUseCase:
    def __init__(self, movie_repository: MovieRepository):
        self.movie_repository = movie_repository

    async def execute(self, movie_id: uuid.UUID, movie_data: MovieSchema) -> MoviePublic:
        existing_movie = await self.movie_repository.get_by_id(movie_id)
        if not existing_movie:
            raise ValueError(f"Movie with id {movie_id} not found")

        title_conflict = await self.movie_repository.get_by_title(movie_data.title)
        if title_conflict and title_conflict.id != movie_id:
            raise ValueError(f"Movie with title '{movie_data.title}' already exists")

        updated_movie = Movie(
            id=movie_id,
            title=movie_data.title,
            genres=movie_data.genres,
            embedding=movie_data.embedding or existing_movie.embedding,
        )

        result = await self.movie_repository.update(updated_movie)

        return MoviePublic(id=result.id, title=result.title, genres=result.genres)
