from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.ports.repositories.model_inference_repository import ModelInferenceRepository
from neural_recommendation.domain.ports.services.logger import LoggerPort
from neural_recommendation.infrastructure.deep_learning.model_inference_manager import ModelInferenceManager


class ModelInferenceManagerAdapter(ModelInferenceRepository):
    """Adapter that implements the port using the existing ModelInferenceManager"""

    def __init__(self, models_dir: str, device: str, data_dir: str, processed_data_dir: str, logger_port: LoggerPort):
        self._manager = ModelInferenceManager(
            logger_port=logger_port,
            models_dir=models_dir,
            device=device,
            data_dir=data_dir,
            processed_data_dir=processed_data_dir,
        )

    def load_model(self) -> NCFModel:
        """Load trained model and feature information"""
        return self._manager.load_model()
