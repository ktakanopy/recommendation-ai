import uuid
from abc import ABC, abstractmethod
from typing import List, Optional

from neural_recommendation.neural_recommendation.domain.models.movie import Movie


class MovieRepository(ABC):
    @abstractmethod
    async def get_by_id(self, movie_id: uuid.UUID) -> Optional[Movie]:
        pass

    @abstractmethod
    async def get_similar_movies(self, query_embedding: List[float], user_watched_movies: List[uuid.UUID], num_recommendations: int) -> List[Movie]:
        pass

    @abstractmethod
    async def get_by_title(self, title: str) -> Optional[Movie]:
        pass

    @abstractmethod
    async def create(self, movie: Movie) -> Movie:
        pass

    @abstractmethod
    async def update(self, movie: Movie) -> Movie:
        pass

    @abstractmethod
    async def delete(self, movie_id: uuid.UUID) -> bool:
        pass
