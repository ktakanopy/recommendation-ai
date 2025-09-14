from typing import Any, Dict, List, Tuple

import torch

from neural_recommendation.applications.services.candidate_generator_service import CandidateGeneratorService
from neural_recommendation.applications.services.ncf_feature_service import NCFFeatureService
from neural_recommendation.domain.exceptions import (
    ColdStartRecommendationError,
)
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from neural_recommendation.domain.ports.services.logger import LoggerPort


class ColdStartRecommender:
    """
    Cold Start Recommendation System for new users

    Handles recommendations for users with limited or no interaction history
    by leveraging user demographics, initial ratings, and content-based filtering.
    """

    def __init__(
        self,
        trained_model: NCFModel,
        movie_features_repository: MovieFeaturesRepository,
        feature_service: NCFFeatureService,
        candidate_generator: CandidateGeneratorService,
        logger: LoggerPort,
        liked_threshold: float = 4.0,
        num_candidates: int = 100,
    ):
        self.model = trained_model
        self.movie_features_repository = movie_features_repository
        self.feature_service = feature_service
        self.candidate_generator = candidate_generator
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.liked_threshold = liked_threshold
        self.num_candidates = num_candidates
        self.logger = logger

    def recommend_for_new_user(
        self,
        user_demographics: Dict[str, Any],
        user_ratings: List[Tuple[int, float]],
        num_recommendations: int = 10,
    ):
        """
        Generate recommendations for a new user

        Args:
            user_demographics: dict with keys 'gender', 'age', 'occupation'
            user_ratings: list of (movie_id, rating) tuples for initial ratings (optional)
            num_recommendations: number of recommendations to return

        Returns:
            list: list of (movie_id, title, predicted_score) tuples
        """

        try:
            user_features = self.feature_service.process_user_demographics(user_demographics)
            user_features = torch.tensor(user_features, dtype=torch.float32)
            user_features = user_features.detach().clone().unsqueeze(0).to(self.device)
        except Exception as e:
            self.logger.error(f"Error processing user demographics: {str(e)}")
            raise e

        candidates = self.candidate_generator.generate_candidates(
            user_demographics,
            user_ratings,
            method="hybrid",
            num_candidates=self.num_candidates,
        )

        if not candidates:
            self.logger.warning("No candidates generated for cold start recommendation")
            return []

        # Score candidates using the NCF model
        try:
            movie_scores = []

            self.model.eval()
            with torch.no_grad():
                for movie_id in candidates:
                    try:
                        # Get movie features
                        movie_features = self.movie_features_repository.get_features(movie_id)
                        movie_features = torch.tensor(movie_features, dtype=torch.float32)
                        movie_features = movie_features.unsqueeze(0).to(self.device)

                        # Predict score using NCF model
                        score = self.model(user_features, movie_features).item()

                        movie_scores.append((movie_id, score))

                    except Exception as e:
                        self.logger.warning(f"Error processing movie {movie_id}: {str(e)}")
                        continue

            # Sort by predicted score and return top recommendations
            movie_scores.sort(key=lambda x: x[1], reverse=True)

            # Filter out movies user has already rated
            rated_movie_ids = {movie_id for movie_id, _ in user_ratings}
            movie_scores = [item for item in movie_scores if item[0] not in rated_movie_ids]

            return movie_scores[:num_recommendations]
        except Exception as e:
            self.logger.error(f"Error generating cold start recommendations: {str(e)}")
            raise ColdStartRecommendationError("Failed to generate cold start recommendations") from e

    def get_onboarding_movies(self, num_movies: int = 10) -> Dict[str, List[int]]:
        """Get onboarding movies for new user"""
        try:
            all_genres = self.movie_features_repository.get_all_genres()
            candidate_ids = self.movie_features_repository.get_top_popular_movies_by_genres(
                all_genres, top_k=num_movies
            )
            return candidate_ids
        except Exception as e:
            self.logger.error(f"Error getting onboarding movies: {str(e)}")
            raise ColdStartRecommendationError(f"Failed to get onboarding movies: {str(e)}")
