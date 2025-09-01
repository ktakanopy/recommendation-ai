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
    def generate_recommendations_for_new_user(
        self, user_age: float, gender: str, preferred_genres: list[str] = None, num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user"""
        pass

    @abstractmethod
    def explain_recommendation(
        self, user_id: str, movie_title: str, user_age: float = 25.0, gender: str = "M"
    ) -> Dict[str, Any]:
        """Explain why a specific movie was recommended"""
        pass
