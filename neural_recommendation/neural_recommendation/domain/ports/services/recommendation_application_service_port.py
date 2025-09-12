from abc import ABC, abstractmethod

from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.models.deep_learning.onboarding_movies import OnboardingMoviesResult


class RecommendationApplicationServicePort(ABC):
    """Port for recommendation generation operations"""

    @abstractmethod
    async def generate_recommendations_cold_start(
        self, user_id: int, num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user using cold start approach"""
        pass

    @abstractmethod
    async def get_onboarding_movies(self, num_movies: int = 10) -> OnboardingMoviesResult:
        """Get onboarding movies for new user"""
        pass
