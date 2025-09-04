from abc import ABC, abstractmethod
from typing import Tuple

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel


class ModelInferenceRepository(ABC):
    """Port for model inference operations"""

    @abstractmethod
    def load_model_and_features(self) -> Tuple[NCFModel, FeatureInfoDto]:
        """Load trained model and feature information"""
        pass
