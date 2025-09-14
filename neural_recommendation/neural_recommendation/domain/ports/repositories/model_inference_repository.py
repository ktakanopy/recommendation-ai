from abc import ABC, abstractmethod

from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel


class ModelInferenceRepository(ABC):
    """Port for model inference operations"""

    @abstractmethod
    def load_model(self) -> NCFModel:
        """Load trained model and feature information"""
        pass
