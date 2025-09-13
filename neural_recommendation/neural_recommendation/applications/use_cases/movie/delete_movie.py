import uuid

from neural_recommendation.applications.interfaces.dtos.message import Message
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository


class DeleteMovieUseCase:
    def __init__(self, movie_repository: MovieRepository):
        self.movie_repository = movie_repository

    async def execute(self, movie_id: uuid.UUID) -> Message:
        existing_movie = await self.movie_repository.get_by_id(movie_id)
        if not existing_movie:
            raise ValueError(f"Movie with id {movie_id} not found")

        success = await self.movie_repository.delete(movie_id)
        if not success:
            raise ValueError(f"Failed to delete movie with id {movie_id}")

        return Message(message=f"Movie '{existing_movie.title}' deleted successfully")
