from typing import Any, Dict, List, Optional, Set

from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.ports.services.recommendation_service_port import RecommendationServicePort


class RecommendationApplicationService(RecommendationServicePort):
    """Application service that implements the recommendation port and handles DTO mapping"""

    def __init__(self, recommendation_generator):
        self._recommendation_generator = recommendation_generator

    def generate_recommendations_for_existing_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        watched_movie_titles: Optional[Set[str]] = None,
        ratings: Optional[List[Rating]] = None,
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for an existing user"""
        return self._recommendation_generator.generate_recommendations_for_existing_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            watched_movie_titles=watched_movie_titles,
            ratings=ratings,
            num_recommendations=num_recommendations
        )

    def generate_recommendations_for_new_user(
        self,
        user_age: float,
        gender: str,
        preferred_genres: List[str] = None,
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user"""
        return self._recommendation_generator.generate_recommendations_for_new_user(
            user_age=user_age,
            gender=gender,
            preferred_genres=preferred_genres,
            num_recommendations=num_recommendations
        )

    def explain_recommendation(
        self,
        user_id: str,
        movie_title: str,
        user_age: float = 25.0,
        gender: str = "M"
    ) -> Dict[str, Any]:
        """Explain why a specific movie was recommended"""
        return self._recommendation_generator.explain_recommendation(
            user_id=user_id,
            movie_title=movie_title,
            user_age=user_age,
            gender=gender
        )
