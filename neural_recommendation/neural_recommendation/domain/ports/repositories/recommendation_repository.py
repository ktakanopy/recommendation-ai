import uuid
from abc import ABC, abstractmethod
from typing import Optional

from neural_recommendation.domain.models.recommendation import Recommendation


class RecommendationRepository(ABC):
    @abstractmethod
    async def get_by_id(self, recommendation_id: uuid.UUID) -> Optional[Recommendation]:
        pass

    @abstractmethod
    async def get_by_user_id(self, user_id: int) -> Optional[Recommendation]:
        pass

    @abstractmethod
    async def create(self, recommendation: Recommendation) -> Recommendation:
        pass

    @abstractmethod
    async def delete(self, recommendation_id: uuid.UUID) -> bool:
        pass
