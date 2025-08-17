from typing import Any, Dict, List, Optional

from two_towers.applications.services.recommendation_orchestrator import RecommendationOrchestrator
from two_towers.domain.models.deep_learning.recommendation import RecommendationResult
from two_towers.domain.ports.repositories.model_inference_repository import ModelInferenceRepository


class RecommendationGenerator:
    """Application use case for generating movie recommendations"""

    def __init__(self, model_repository: ModelInferenceRepository):
        self.model_repository = model_repository
        self._recommendation_orchestrator = None

    def _get_recommendation_orchestrator(self) -> RecommendationOrchestrator:
        """Lazy load recommendation orchestrator"""
        if self._recommendation_orchestrator is None:
            model, feature_info = self.model_repository.load_model_and_features()
            self._recommendation_orchestrator = RecommendationOrchestrator(model, feature_info)
        return self._recommendation_orchestrator

    def generate_recommendations_for_existing_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """
        Generate recommendations for an existing user
        
        Args:
            user_id: User identifier
            user_age: User's age for demographic features
            gender: User's gender ("M" or "F")
            watched_movie_titles: Set of movie titles the user has already watched
            num_recommendations: Number of recommendations to generate
            
        Returns:
            RecommendationResult containing the recommendations
        """

        recommendation_orchestrator = self._get_recommendation_orchestrator()

        return recommendation_orchestrator.generate_recommendations_for_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=num_recommendations
        )

    def generate_recommendations_for_new_user(
        self,
        user_age: float,
        gender: str,
        preferred_genres: Optional[List[str]] = None,
        num_recommendations: int = 10
    ) -> RecommendationResult:
        """
        Generate recommendations for a new user based on demographics
        
        Args:
            user_age: User's age
            gender: User's gender ("M" or "F")
            preferred_genres: Optional list of preferred genres
            num_recommendations: Number of recommendations to generate
            
        Returns:
            RecommendationResult containing the recommendations
        """

        # For new users, use a default user_id
        temp_user_id = "new_user"

        recommendation_orchestrator = self._get_recommendation_orchestrator()

        # TODO: In the future, you could implement genre-based filtering here
        # For now, generate recommendations based on demographics only

        return recommendation_orchestrator.generate_recommendations_for_user(
            user_id=temp_user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=num_recommendations
        )

    def explain_recommendation(
        self,
        user_id: str,
        movie_title: str,
        user_age: float = 25.0,
        gender: str = "M"
    ) -> Dict[str, Any]:
        """
        Provide explanation for why a specific movie was recommended
        
        Args:
            user_id: User identifier  
            movie_title: Title of the movie to explain
            user_age: User's age
            gender: User's gender
            
        Returns:
            Dictionary with explanation details
        """

        recommendation_orchestrator = self._get_recommendation_orchestrator()

        # Generate single recommendation for this movie
        result = recommendation_orchestrator.generate_recommendations_for_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=1000  # Get many to find this specific movie
        )

        # Find the specific movie in recommendations
        target_recommendation = None
        for rec in result.recommendations:
            if rec.title == movie_title:
                target_recommendation = rec
                break

        if target_recommendation is None:
            return {
                "movie_title": movie_title,
                "explanation": "Movie not found in recommendation candidates",
                "similarity_score": 0.0
            }

        return {
            "movie_title": movie_title,
            "similarity_score": target_recommendation.similarity_score,
            "similarity_percentage": target_recommendation.similarity_percentage,
            "explanation": f"This movie has a {target_recommendation.similarity_percentage:.1f}% similarity match with your preferences based on your viewing history and demographic profile.",
            "genres": target_recommendation.genres
        }
