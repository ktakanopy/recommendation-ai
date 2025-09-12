from abc import ABC, abstractmethod
from typing import List, Optional

from neural_recommendation.domain.models.movie import Movie


class MovieRepository(ABC):
    @abstractmethod
    async def get_by_id(self, movie_id: int) -> Optional[Movie]:
        pass

    @abstractmethod
    async def get_by_title(self, title: str) -> Optional[Movie]:
        pass

    @abstractmethod
    async def get_all(self, offset: int = 0, limit: int = 100) -> List[Movie]:
        pass

    @abstractmethod
    async def create(self, movie: Movie) -> Movie:
        pass

    @abstractmethod
    async def update(self, movie: Movie) -> Movie:
        pass

    @abstractmethod
    async def delete(self, movie_id: int) -> bool:
        pass
