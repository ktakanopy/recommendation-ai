from neural_recommendation.applications.services.candidate_generator_service import CandidateGeneratorService
from neural_recommendation.applications.services.ncf_feature_service import NCFFeatureService
from neural_recommendation.domain.models.deep_learning.onboarding_movies import OnboardingMoviesResult
from neural_recommendation.domain.models.deep_learning.recommendation import RecommendationResult
from neural_recommendation.domain.ports.repositories.feature_encoder_repository import FeatureEncoderRepository
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository
from neural_recommendation.domain.ports.repositories.user_features_repository import UserFeaturesRepository
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.domain.ports.services.recommendation_application_service_port import (
    RecommendationApplicationServicePort,
)
from neural_recommendation.domain.services.recommendation_service import RecommendationService
from neural_recommendation.infrastructure.config.settings import MLModelSettings
from neural_recommendation.domain.ports.services.logger import LoggerPort


class RecommendationApplicationService(RecommendationApplicationServicePort):
    """Application service for NCF-based recommendations"""

    def __init__(
        self,
        ml_settings: MLModelSettings,
        model_repository: ModelInferenceRepository,
        movie_repository: MovieRepository,
        user_repository: UserRepository,
        rating_repository: RatingRepository,
        feature_encoder_repository: FeatureEncoderRepository,
        user_features_repository: UserFeaturesRepository,
        movie_features_repository: MovieFeaturesRepository,
        logger: LoggerPort,
    ):
        self.ml_settings = ml_settings
        self._model_repository = model_repository
        self.user_repository = user_repository
        self.rating_repository = rating_repository
        self.feature_encoder_repository = feature_encoder_repository
        self.user_features_repository = user_features_repository
        self.movie_repository = movie_repository
        self.movie_features_repository = movie_features_repository
        self.logger = logger
        self._domain_service = self._get_domain_service()

    def _get_domain_service(self) -> RecommendationService:
        """Initialize the domain service with NCF model and feature processor"""
        self.logger.info("Initializing NCF-based recommendation service")

        # Load NCF model
        model = self._model_repository.load_model()
        # Initialize NCF feature processor
        feature_service = NCFFeatureService(feature_encoder_repository=self.feature_encoder_repository, logger=self.logger)

        self.candidate_generator = CandidateGeneratorService(
            movie_features_repository=self.movie_features_repository,
            user_features_repository=self.user_features_repository,
            feature_service=feature_service,
            logger=self.logger,
        )
        # Create domain service
        domain_service = RecommendationService(
            model=model,
            feature_service=feature_service,
            candidate_generator=self.candidate_generator,
            movie_repository=self.movie_repository,
            movie_features_repository=self.movie_features_repository,
            logger=self.logger,
        )

        return domain_service

    async def generate_recommendations_cold_start(
        self, user_id: int, num_recommendations: int = 10
    ) -> RecommendationResult:
        """Generate recommendations for a new user using NCF cold start approach"""
        self.logger.info(f"Generating cold start recommendations for user: {user_id}")

        # Get user from repository
        user = await self.user_repository.get_by_id(user_id)
        if not user:
            raise ValueError(f"User with ID {user_id} not found")

        ratings = await self.rating_repository.get_by_user_id(user.id)
        user.ratings = ratings

        # Use the specialized cold start method with user data from database
        return await self._domain_service.generate_recommendations_cold_start(
            user=user,
            num_recommendations=num_recommendations,
        )

    async def get_onboarding_movies(self, num_movies: int = 10) -> OnboardingMoviesResult:
        """Get onboarding movies for new user"""
        return await self._domain_service.get_onboarding_movies(num_movies=num_movies)
