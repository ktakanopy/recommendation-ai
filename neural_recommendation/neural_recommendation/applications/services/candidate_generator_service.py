from collections import defaultdict
from typing import Any, Dict, List, Tuple

from neural_recommendation.applications.services.ncf_feature_service import NCFFeatureService
from neural_recommendation.domain.exceptions import CandidateGeneratorError, ValidationError
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from neural_recommendation.domain.ports.repositories.user_features_repository import UserFeaturesRepository
from neural_recommendation.domain.ports.services.logger import LoggerPort


# global data structures for fast candidate generation
class CandidateGeneratorService:
    def __init__(
        self,
        movie_features_repository: MovieFeaturesRepository,
        user_features_repository: UserFeaturesRepository,
        feature_service: NCFFeatureService,
        logger: LoggerPort,
    ):
        self.movie_features_repository = movie_features_repository
        self.user_features_repository = user_features_repository
        self.feature_processor = feature_service
        self.logger = logger

    def _removed_rated_items(self, available_items: List[int], user_ratings: List[Tuple[int, float]]):
        """Remove items that the user has already rated"""
        user_movies = [item for item, _ in user_ratings]
        return [item for item in available_items if item not in user_movies]

    def _generate_popularity_candidates(self, user_ratings: List[Tuple[int, float]], num_candidates=100):
        """Optimized popularity-based candidate generation"""
        available_items = set(self.movie_features_repository.get_top_popular_movies(num_candidates))

        # remove items that the user has already rated
        available_items = self._removed_rated_items(available_items, user_ratings)
        return available_items

    def _generate_collaborative_candidates(
        self, user_demographics: Dict[str, Any], user_ratings: List[Tuple[int, float]], num_candidates=100
    ):
        """Optimized collaborative filtering using pre-computed similarity"""
        user_feature = self.feature_processor.process_user_demographics(user_demographics)
        similar_user_ids = self.user_features_repository.get_similar_users(user_feature, top_k=num_candidates)

        candidate_scores = defaultdict(float)
        for similar_user_id in similar_user_ids:
            candidate_scores[similar_user_id] += 1.0

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        available_items = [item for item, score in sorted_candidates[:num_candidates]]
        available_items = self._removed_rated_items(available_items, user_ratings)
        return available_items

    def _generate_content_candidates(
        self,
        user_ratings: List[Tuple[int, float]],
        num_candidates=100,
    ):
        if not isinstance(user_ratings, list):
            raise ValidationError("user_ratings must be a list of tuples")
        interacted = [item for item, _ in user_ratings]
        candidate_scores = defaultdict(float)
        for item in interacted:
            try:
                similars = self.movie_features_repository.get_similar_movies(item, top_k=num_candidates)
            except Exception:
                similars = []
            for m in similars:
                candidate_scores[m] += 1.0

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        available_items = [item for item, score in sorted_candidates[:num_candidates]]
        available_items = self._removed_rated_items(available_items, user_ratings)
        return available_items

    def _generate_hybrid_candidates(
        self, user_demographics: Dict[str, Any], user_ratings: List[Tuple[int, float]], num_candidates=100
    ):
        """Optimized hybrid candidate generation"""
        pop_candidates = self._generate_popularity_candidates(user_ratings, num_candidates=num_candidates // 3)
        collab_candidates = self._generate_collaborative_candidates(
            user_demographics, user_ratings, num_candidates=num_candidates // 3
        )
        content_candidates = self._generate_content_candidates(user_ratings, num_candidates=num_candidates // 3)
        hybrid_candidates = list(set(pop_candidates + collab_candidates + content_candidates))

        return hybrid_candidates[:num_candidates]

    def generate_candidates(
        self, user_demographics: Dict[str, Any], ratings: List[Tuple[int, float]], method="hybrid", num_candidates=100
    ):
        """Main interface for candidate generation"""
        try:
            if method == "popularity":
                return self._generate_popularity_candidates(ratings, num_candidates=num_candidates)
            elif method == "collaborative":
                return self._generate_collaborative_candidates(
                    user_demographics, ratings, num_candidates=num_candidates
                )
            elif method == "content":
                return self._generate_content_candidates(ratings, num_candidates=num_candidates)
            elif method == "hybrid":
                return self._generate_hybrid_candidates(user_demographics, ratings, num_candidates=num_candidates)
            else:
                raise ValueError(f"Invalid method: {method}")
        except Exception as e:
            raise CandidateGeneratorError(f"Error generating candidates: {str(e)}") from e
