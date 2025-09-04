import uuid

from neural_recommendation.applications.interfaces.schemas import MoviePublic
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository


class GetMovieUseCase:
    def __init__(self, movie_repository: MovieRepository):
        self.movie_repository = movie_repository

    async def execute(self, movie_id: uuid.UUID) -> MoviePublic:
        movie = await self.movie_repository.get_by_id(movie_id)
        if not movie:
            raise ValueError(f"Movie with id {movie_id} not found")

        return MoviePublic(id=movie.id, title=movie.title, genres=movie.genres)
