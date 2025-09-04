import os
from typing import Any, Dict

from neural_recommendation.domain.models.deep_learning.model_config import ModelConfig
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ModelLoader:
    def __init__(self, model_path: str):
        self.model_path = model_path

    @staticmethod
    def load_model(config: ModelConfig, additional_feature_info: Dict[str, Any]) -> NCFModel:
        """Load NCF model based on configuration"""
        model_path = os.path.join(config.models_dir, config.model_name)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        return ModelLoader._load_ncf_model(config, model_path)

    @staticmethod
    def _load_ncf_model(config: ModelConfig, model_path: str) -> NCFModel:
        """Load NCF model from checkpoint"""
        logger.info(f"Loading NCF model from {model_path}")

        try:
            # Try to load using the class method first
            model = NCFModel.load_model(filepath=model_path, num_negatives=getattr(config, "num_negatives", 4))
            model.to(config.device)
            logger.info("NCF model loaded successfully using class method")
            return model

        except Exception as e:
            raise e
