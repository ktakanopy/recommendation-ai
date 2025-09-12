from collections import defaultdict
from typing import Any, Dict, List

from neural_recommendation.applications.services.candidate_generator_service import CandidateGeneratorService
from neural_recommendation.applications.services.ncf_feature_service import NCFFeatureService
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.domain.models.deep_learning.onboarding_movies import OnboardingMovie, OnboardingMoviesResult
import torch

from neural_recommendation.applications.use_cases.deep_learning.cold_start_recommender import ColdStartRecommender
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.domain.models.user import User
from neural_recommendation.infrastructure.logging.logger import (
    Logger,
)  # TODO: use the logger from the application layer

logger = Logger.get_logger(__name__)


class RecommendationService:
    """Domain service for generating movie recommendations using NCF model"""

    def __init__(
        self,
        model: NCFModel,
        feature_service: NCFFeatureService,
        candidate_generator: CandidateGeneratorService,
        movie_repository: MovieRepository,
        movie_features_repository: MovieFeaturesRepository,
    ):
        self.model = model
        self.feature_service = feature_service
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.cold_start_recommender = ColdStartRecommender(
            trained_model=model,
            feature_service=feature_service,
            candidate_generator=candidate_generator,
            liked_threshold=4.0,
            movie_features_repository=movie_features_repository,
        )
        self.movie_repository = movie_repository

    async def generate_recommendations_cold_start(
        self,
        user: User,
        num_recommendations: int = 10,
    ) -> RecommendationResult:
        """Generate recommendations for a new user using cold start approach"""
        logger.info(f"Generating cold start recommendations for user: {user.username} (ID: {user.id})")

        # Prepare user demographics from User object
        user_demographics = {"gender": user.gender, "age": user.age, "occupation": user.occupation}

        # Convert user ratings to tuples for cold start recommender
        user_ratings = None
        if user.ratings:
            user_ratings = [(rating.movie_id, rating.rating) for rating in user.ratings]
            logger.info(f"User has {len(user_ratings)} existing ratings")
        else:
            raise ValueError("User has no ratings")

        # Use cold start recommender
        try:
            cold_start_results = self.cold_start_recommender.recommend_for_new_user(
                user_demographics=user_demographics, user_ratings=user_ratings, num_recommendations=num_recommendations
            )
            # Convert to Recommendation objects
            recommendations = []
            for i, (movie_id, score) in enumerate(cold_start_results):
                movie = await self.movie_repository.get_by_id(movie_id)
                if not movie:
                    logger.warning(f"Movie with id {movie_id} not found in movie repository")
                    continue

                recommendation = Recommendation(
                    movie_id=movie_id,
                    title=movie.title,
                    genres=movie.genres,
                    similarity_score=float(score),
                )
                recommendations.append(recommendation)

            return RecommendationResult(user_id=str(user.id), recommendations=recommendations)

        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {str(e)}")
            raise e

    async def get_onboarding_movies(self, num_movies: int = 10) -> List[Dict[str, Any]]:
        """Get diverse movies for new user onboarding"""
        candidates = self.cold_start_recommender.get_onboarding_movies(num_movies=num_movies)
        recommendations = defaultdict(list)
        for genre, movie_ids in candidates.items():
            for movie_id in movie_ids:
                movie = await self.movie_repository.get_by_id(movie_id)
                if not movie:
                    logger.warning(f"Movie with id {movie_id} not found in movie repository")
                    continue
                recommendations[genre].append(
                    OnboardingMovie(movie_id=movie_id, title=movie.title, genres=movie.genres)
                )
        return OnboardingMoviesResult(recommendations=recommendations)
