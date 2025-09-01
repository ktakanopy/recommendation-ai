from typing import Tuple

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.domain.models.deep_learning.two_tower_model import TwoTowerModel
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.infrastructure.deep_learning.model_inference_manager import ModelInferenceManager


class ModelInferenceManagerAdapter(ModelInferenceRepository):
    """Adapter that implements the port using the existing ModelInferenceManager"""

    def __init__(self, models_dir: str, device: str):
        self._manager = ModelInferenceManager(models_dir=models_dir, device=device)

    def load_model_and_features(self) -> Tuple[TwoTowerModel, FeatureInfoDto]:
        """Load trained model and feature information"""
        return self._manager.load_model_and_features()
