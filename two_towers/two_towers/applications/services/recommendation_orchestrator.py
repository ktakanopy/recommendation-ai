from typing import Any, Dict

from two_towers.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from two_towers.applications.use_cases.deep_learning.feature_preparation_service import FeaturePreparationService
from two_towers.domain.models.deep_learning.recommendation import RecommendationResult
from two_towers.domain.models.deep_learning.two_tower_model import TwoTowerModel
from two_towers.domain.services.recommendation_service import RecommendationService
from two_towers.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class RecommendationOrchestrator:
    """Application service for orchestrating movie recommendations"""

    def __init__(self, model: TwoTowerModel, feature_info: FeatureInfoDto):
        self.model = model
        self.feature_info = feature_info
        self.feature_service = FeaturePreparationService(feature_info)

        # Setup movie mappings for domain service
        self.movie_mappings = self._setup_movie_mappings()

        # Create domain service
        self.domain_service = RecommendationService(
            model=model,
            feature_service=self.feature_service,
            movie_mappings=self.movie_mappings
        )

    def _setup_movie_mappings(self) -> Dict[str, Any]:
        """Setup movie title to index mappings"""
        title_to_idx = self.feature_info.sentence_embeddings.title_to_idx
        idx_to_title = {idx: title for title, idx in title_to_idx.items()}
        all_movie_titles = list(title_to_idx.keys())

        return {
            "title_to_idx": title_to_idx,
            "idx_to_title": idx_to_title,
            "all_movie_titles": all_movie_titles
        }

    def generate_recommendations_for_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        num_recommendations: int = 10,
        batch_size: int = 100,
    ) -> RecommendationResult:
        """Orchestrate recommendation generation for a user"""
        logger.info(f"Orchestrating recommendation generation for user {user_id}")
        return self.domain_service.generate_recommendations_for_user(
            user_id=user_id,
            user_age=user_age,
            gender=gender,
            num_recommendations=num_recommendations,
            batch_size=batch_size
        )
