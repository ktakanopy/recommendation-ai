from abc import ABC, abstractmethod
from typing import List

from neural_recommendation.domain.models.rating import Rating


class RatingRepository(ABC):
    @abstractmethod
    async def bulk_create(self, user_id: int, ratings: List[Rating]) -> None:
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: int) -> List[Rating]:
        pass

