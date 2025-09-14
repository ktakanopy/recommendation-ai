from collections import defaultdict
from typing import Any, Dict, List

import torch

from neural_recommendation.applications.services.candidate_generator_service import CandidateGeneratorService
from neural_recommendation.applications.services.ncf_feature_service import NCFFeatureService
from neural_recommendation.applications.use_cases.deep_learning.cold_start_recommender import ColdStartRecommender
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.models.deep_learning.onboarding_movies import OnboardingMovie, OnboardingMoviesResult
from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.domain.models.user import User
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.domain.ports.services.logger import LoggerPort


class RecommendationService:
    def __init__(
        self,
        model: NCFModel,
        feature_service: NCFFeatureService,
        candidate_generator: CandidateGeneratorService,
        movie_repository: MovieRepository,
        movie_features_repository: MovieFeaturesRepository,
        logger: LoggerPort,
    ):
        self.model = model
        self.feature_service = feature_service
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.eval()
        self.cold_start_recommender = ColdStartRecommender(
            trained_model=model,
            movie_features_repository=movie_features_repository,
            feature_service=feature_service,
            candidate_generator=candidate_generator,
            logger=logger,
            liked_threshold=4.0,
            num_candidates=100,
        )
        self.movie_repository = movie_repository
        self.logger = logger

    async def generate_recommendations_cold_start(
        self,
        user: User,
        num_recommendations: int = 10,
    ) -> RecommendationResult:
        self.logger.info(f"Generating cold start recommendations for user: {user.username} (ID: {user.id})")

        user_demographics = {"gender": user.gender, "age": user.age, "occupation": user.occupation}

        user_ratings = []
        if user.ratings:
            user_ratings = [(rating.movie_id, rating.rating) for rating in user.ratings]
            self.logger.info(f"User has {len(user_ratings)} existing ratings")
        else:
            self.logger.warning("User has no ratings, cold start recommender will use default values.")

        try:
            cold_start_results = self.cold_start_recommender.recommend_for_new_user(
                user_demographics=user_demographics, user_ratings=user_ratings, num_recommendations=num_recommendations
            )
            recommendations = []
            for movie_id, score in cold_start_results:
                movie = await self.movie_repository.get_by_id(movie_id)
                if not movie:
                    self.logger.warning(f"Movie with id {movie_id} not found in movie repository")
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
            self.logger.error(f"Error generating cold start recommendations: {str(e)}")
            raise e

    async def get_onboarding_movies(self, num_movies: int = 10) -> List[Dict[str, Any]]:
        candidates = self.cold_start_recommender.get_onboarding_movies(num_movies=num_movies)
        recommendations = defaultdict(list)
        for genre, movie_ids in candidates.items():
            for movie_id in movie_ids:
                movie = await self.movie_repository.get_by_id(movie_id)
                if not movie:
                    self.logger.warning(f"Movie with id {movie_id} not found in movie repository")
                    continue
                recommendations[genre].append(
                    OnboardingMovie(movie_id=movie_id, title=movie.title, genres=movie.genres)
                )
        return OnboardingMoviesResult(recommendations=recommendations)


