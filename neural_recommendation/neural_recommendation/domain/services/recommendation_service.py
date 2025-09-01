from typing import Any, Dict, List

import numpy as np
import torch
import torch.nn.functional as F

from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.domain.models.deep_learning.two_tower_model import TwoTowerModel
from neural_recommendation.infrastructure.logging.logger import Logger
from neural_recommendation.applications.use_cases.deep_learning.feature_preparation_service import (
    FeaturePreparationService,
)

logger = Logger.get_logger(__name__)


class RecommendationService:
    """Domain service for generating movie recommendations"""

    def __init__(self, model: TwoTowerModel, feature_service: FeaturePreparationService, movie_mappings: Dict[str, Any]):
        self.model = model
        self.feature_service = feature_service
        self.device = next(model.parameters()).device
        self.model.eval()
        self.title_to_idx = movie_mappings.get("title_to_idx", {})
        self.idx_to_title = movie_mappings.get("idx_to_title", {})
        self.all_movie_titles = movie_mappings.get("all_movie_titles", [])

    def generate_recommendations_for_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        num_recommendations: int = 10,
        batch_size: int = 100,
    ) -> RecommendationResult:
        logger.info(f"Generating recommendations for user {user_id}")
        available_movie_titles = self.all_movie_titles
        user_features = self.feature_service.prepare_user_features(
            user_id=user_id, ratings=[], user_age=user_age, gender=gender
        )
        user_embedding = self._get_user_embedding(user_features)
        similarities = self._calculate_movie_similarities(user_embedding, available_movie_titles, batch_size)
        recommendations = self._create_top_recommendations(available_movie_titles, similarities, num_recommendations)
        return RecommendationResult(
            user_id=user_id, recommendations=recommendations, total_available_movies=len(available_movie_titles)
        )

    def _get_user_embedding(self, user_features: Dict[str, Any]) -> torch.Tensor:
        """Get normalized user embedding from features"""
        # Convert features to tensors
        user_inputs = {}
        for key, value in user_features.items():
            if isinstance(value, (int, float)):
                user_inputs[key] = torch.tensor([value], device=self.device, dtype=torch.float32)
            else:
                user_inputs[key] = torch.tensor([value], device=self.device)

        # Handle user_id as long tensor
        if "user_id" in user_inputs:
            user_inputs["user_id"] = user_inputs["user_id"].long()

        # Get user embedding
        with torch.no_grad():
            user_embedding = self.model.user_model(user_inputs)
            query_embedding = self.model.query_tower(user_embedding)
            return F.normalize(query_embedding, p=2, dim=1)

    def _calculate_movie_similarities(
        self, user_embedding: torch.Tensor, available_movies: List[str], batch_size: int
    ) -> List[float]:
        """Calculate similarities between user and movies in batches"""
        similarities = []

        for i in range(0, len(available_movies), batch_size):
            batch_movies = available_movies[i : i + batch_size]
            batch_similarities = self._process_movie_batch(user_embedding, batch_movies)
            similarities.extend(batch_similarities)

        return similarities

    def _process_movie_batch(self, user_embedding: torch.Tensor, movie_titles: List[str]) -> List[float]:
        """Process a batch of movies and return similarities"""
        # Get movie indices
        movie_indices = [self.title_to_idx.get(title, 0) for title in movie_titles]

        # Prepare movie inputs
        movie_inputs = {"movie_idx": torch.tensor(movie_indices, device=self.device, dtype=torch.long)}

        # Get movie embeddings
        with torch.no_grad():
            movie_embeddings = self.model.movie_model(movie_inputs)
            candidate_embeddings = self.model.candidate_tower(movie_embeddings)
            candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)

            # Calculate similarities
            batch_similarities = torch.matmul(user_embedding, candidate_embeddings.T)
            return batch_similarities.cpu().numpy().flatten().tolist()

    def _create_top_recommendations(
        self, available_movies: List[str], similarities: List[float], num_recommendations: int
    ) -> List[Recommendation]:
        """Create top N recommendations from similarities"""
        # Get top indices
        top_indices = np.argsort(similarities)[::-1][:num_recommendations]

        recommendations = []
        for idx in top_indices:
            movie_title = available_movies[idx]
            similarity_score = similarities[idx]

            # Extract movie info (you might want to get this from a movie repository)
            # For now, we'll use placeholder values
            movie_id = self.title_to_idx.get(movie_title, 0)

            recommendation = Recommendation(
                movie_id=movie_id,
                title=movie_title,
                genres="Unknown",  # TODO: Get from movie repository
                similarity_score=similarity_score,
            )
            recommendations.append(recommendation)

        return recommendations
