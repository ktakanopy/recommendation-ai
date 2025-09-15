import os
from typing import Tuple

import torch

from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.ports.services.logger import LoggerPort
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ModelInferenceManager:
    """Infrastructure service for managing NCF model inference"""

    def __init__(
        self,
        logger_port: LoggerPort,
        models_dir: str = "models",
        device: str = "cpu",
        data_dir: str = "data",
        processed_data_dir: str = "data/processed_data",
    ):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.processed_data_dir = processed_data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.logger_port = logger_port
        self._model = None

    def load_model(
        self,
        model_filename: str = "ncf_model.pth",
        features_filename: str = "preprocessed_features.pkl",
    ) -> Tuple[NCFModel]:
        """Load trained NCF model and feature information"""

        if self._model is not None:
            return self._model

        logger.info("Loading NCF model and features for inference...")

        # Load NCF model
        self._model = NCFModel.load_model(os.path.join(self.models_dir, model_filename), logger_port=self.logger_port)

        logger.info(f"NCF model and features loaded successfully on device: {self.device}")
        return self._model
