from neural_recommendation.infrastructure.config.settings import MLModelSettings

from neural_recommendation.applications.use_cases.deep_learning.ncf_feature_processor import NCFFeatureProcessor
from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.recommendation_application_service_port import (
    RecommendationApplicationServicePort,
)
from neural_recommendation.domain.services.recommendation_service import RecommendationService
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class RecommendationApplicationService(RecommendationApplicationServicePort):
    """Application service for NCF-based recommendations"""

    def __init__(self, ml_settings: MLModelSettings, model_repository: ModelInferenceRepository, user_repository: UserRepository, rating_repository: RatingRepository):
        self.ml_settings = ml_settings
        self._model_repository = model_repository
        self.user_repository = user_repository
        self.rating_repository = rating_repository
        self._domain_service = self._get_domain_service()

    def _get_domain_service(self) -> RecommendationService:
        """Initialize the domain service with NCF model and feature processor"""
        logger.info("Initializing NCF-based recommendation service")

        # Load NCF model
        model, feature_info = self._model_repository.load_model_and_features()
        # Initialize NCF feature processor
        feature_service = NCFFeatureProcessor()

        # Create movie mappings - for NCF we'll use a simplified approach
        title_to_idx = feature_info.sentence_embeddings.title_to_idx

        idx_to_title = {idx: title for title, idx in title_to_idx.items()}
        all_movie_titles = list(title_to_idx.keys())

        movie_mappings = {
            "title_to_idx": title_to_idx,
            "idx_to_title": idx_to_title,
            "all_movie_titles": all_movie_titles,
        }

        # Create domain service
        domain_service = RecommendationService(
            model=model, feature_service=feature_service, movie_mappings=movie_mappings
        )

        logger.info(f"NCF recommendation service initialized with {len(all_movie_titles)} movies")

        return domain_service

    async def generate_recommendations_cold_start(
        self, user_id: int, num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user using NCF cold start approach"""
        logger.info(f"Generating cold start recommendations for user: {user_id}")

        # Get user from repository
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        ratings = await self.rating_repository.get_by_user_id(user.id)
        user.ratings = ratings

        # Use the specialized cold start method with user data from database
        return self._domain_service.generate_recommendations_cold_start(
            user=user,
            num_recommendations=num_recommendations,
        )
