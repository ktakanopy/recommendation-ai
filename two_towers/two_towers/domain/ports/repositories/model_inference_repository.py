from abc import ABC, abstractmethod
from typing import Tuple

from two_towers.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from two_towers.domain.models.deep_learning.two_tower_model import TwoTowerModel


class ModelInferenceRepository(ABC):
    """Port for model inference operations"""

    @abstractmethod
    def load_model_and_features(self) -> Tuple[TwoTowerModel, FeatureInfoDto]:
        """Load trained model and feature information"""
        pass
