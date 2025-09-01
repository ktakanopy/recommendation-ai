from abc import ABC, abstractmethod
from typing import Any, Dict

from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult


class RecommendationServicePort(ABC):
    """Port for recommendation generation operations"""

    @abstractmethod
    def generate_recommendations_for_existing_user(
        self, user_id: str, user_age: float = 25.0, gender: str = "M", num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for an existing user"""
        pass

    @abstractmethod
    async def generate_recommendations_cold_start(
        self, user_id: int, num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user using cold start approach"""
        pass