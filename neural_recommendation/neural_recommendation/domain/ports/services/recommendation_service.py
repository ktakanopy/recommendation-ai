import uuid
from abc import ABC, abstractmethod
from typing import List

from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation


class RecommendationService(ABC):
    @abstractmethod
    def get_recommendations(self, user_id: uuid.UUID) -> List[Recommendation]:
        pass

    @abstractmethod
    def generate_recommendations(self, user_id: uuid.UUID) -> List[Recommendation]:
        pass
