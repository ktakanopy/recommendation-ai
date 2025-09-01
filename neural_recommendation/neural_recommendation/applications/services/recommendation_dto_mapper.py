from typing import Any, Dict, List

from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.models.rating import Rating


class RecommendationDtoMapper:
    """Service for mapping between domain models and DTOs"""

    @staticmethod
    def to_recommendation_response_dict(recommendation) -> Dict[str, Any]:
        """Convert domain Recommendation to response dict"""
        return {
            "movie_id": recommendation.movie_id,
            "title": recommendation.title,
            "genres": recommendation.genres,
            "similarity_score": recommendation.similarity_score,
            "similarity_percentage": recommendation.similarity_percentage
        }

    @staticmethod
    def to_recommendation_result_response_dict(result: RecommendationResult) -> Dict[str, Any]:
        """Convert domain RecommendationResult to response dict"""
        recommendations = [
            RecommendationDtoMapper.to_recommendation_response_dict(rec)
            for rec in result.recommendations
        ]

        return {
            "user_id": result.user_id,
            "recommendations": recommendations,
            "total_available_movies": result.total_available_movies,
            "recommendation_count": result.recommendation_count
        }

    @staticmethod
    def to_explanation_response_dict(explanation: Dict[str, Any]) -> Dict[str, Any]:
        """Convert explanation dict to response dict"""
        return {
            "movie_title": explanation["movie_title"],
            "similarity_score": explanation["similarity_score"],
            "similarity_percentage": explanation.get("similarity_percentage", 0.0),
            "explanation": explanation["explanation"],
            "genres": explanation.get("genres", "Unknown")
        }

    @staticmethod
    def rating_request_to_domain(rating_request) -> Rating:
        """Convert RatingRequest DTO to domain Rating"""
        return Rating(
            id=rating_request.id,
            user_id=rating_request.user_id,
            movie_id=rating_request.movie_id,
            timestamp=rating_request.timestamp,
            rating=rating_request.rating
        )

    @staticmethod
    def rating_requests_to_domain(rating_requests: List) -> List[Rating]:
        """Convert list of RatingRequest DTOs to domain Rating objects"""
        if not rating_requests:
            return []

        return [
            RecommendationDtoMapper.rating_request_to_domain(rating_req)
            for rating_req in rating_requests
        ]
