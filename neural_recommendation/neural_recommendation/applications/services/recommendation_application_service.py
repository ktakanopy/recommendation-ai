from typing import Any, Dict

from neural_recommendation.applications.use_cases.deep_learning.feature_preparation_service import (
    FeaturePreparationService,
)
from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.services.recommendation_service_port import RecommendationServicePort
from neural_recommendation.domain.services.recommendation_service import RecommendationService


class RecommendationApplicationService(RecommendationServicePort):
    def __init__(self, model_repository: ModelInferenceRepository):
        self._model_repository = model_repository
        self._domain_service = None

    def _get_domain_service(self) -> RecommendationService:
        if self._domain_service is None:
            model, feature_info = self._model_repository.load_model_and_features()
            feature_service = FeaturePreparationService(feature_info)
            title_to_idx = feature_info.sentence_embeddings.title_to_idx
            idx_to_title = {idx: title for title, idx in title_to_idx.items()}
            all_movie_titles = list(title_to_idx.keys())
            movie_mappings = {
                "title_to_idx": title_to_idx,
                "idx_to_title": idx_to_title,
                "all_movie_titles": all_movie_titles,
            }
            self._domain_service = RecommendationService(model=model, feature_service=feature_service, movie_mappings=movie_mappings)
        return self._domain_service

    def generate_recommendations_for_existing_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        num_recommendations: int = 10
    ) -> RecommendationResult:
        domain_service = self._get_domain_service()
        return domain_service.generate_recommendations_for_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=num_recommendations,
        )

    def generate_recommendations_for_new_user(
        self,
        user_age: float,
        gender: str,
        preferred_genres: list[str] = None,
        num_recommendations: int = 10
    ) -> RecommendationResult:
        domain_service = self._get_domain_service()
        return domain_service.generate_recommendations_for_user(
            user_id="new_user",
            user_age=user_age,
            gender=gender,
            num_recommendations=num_recommendations,
        )

    def explain_recommendation(
        self,
        user_id: str,
        movie_title: str,
        user_age: float = 25.0,
        gender: str = "M"
    ) -> Dict[str, Any]:
        domain_service = self._get_domain_service()
        result = domain_service.generate_recommendations_for_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=1000,
        )
        target = None
        for rec in result.recommendations:
            if rec.title == movie_title:
                target = rec
                break
        if target is None:
            return {
                "movie_title": movie_title,
                "explanation": "Movie not found in recommendation candidates",
                "similarity_score": 0.0,
                "similarity_percentage": 0.0,
                "genres": "Unknown",
            }
        return {
            "movie_title": movie_title,
            "similarity_score": target.similarity_score,
            "similarity_percentage": target.similarity_percentage,
            "explanation": f"This movie has a {target.similarity_percentage:.1f}% similarity match with your preferences based on your viewing history and demographic profile.",
            "genres": target.genres,
        }
