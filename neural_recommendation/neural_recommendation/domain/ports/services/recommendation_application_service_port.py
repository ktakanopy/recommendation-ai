from abc import ABC, abstractmethod

from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult


class RecommendationApplicationServicePort(ABC):
    """Port for recommendation generation operations"""
    @abstractmethod
    async def generate_recommendations_cold_start(
        self, user_id: int, num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user using cold start approach"""
        pass
