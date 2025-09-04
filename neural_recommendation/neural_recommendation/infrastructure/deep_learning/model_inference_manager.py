import os
from typing import Tuple

import torch

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ModelInferenceManager:
    """Infrastructure service for managing NCF model inference"""

    def __init__(self, models_dir: str = "models", device: str = "cpu", data_dir: str = "data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._feature_info = None

    def load_model_and_features(
        self,
        model_filename: str = "ncf_model.pth",
        features_filename: str = "preprocessed_features.pkl",
    ) -> Tuple[NCFModel, FeatureInfoDto]:
        """Load trained NCF model and feature information"""

        if self._model is not None and self._feature_info is not None:
            return self._model, self._feature_info

        logger.info("Loading NCF model and features for inference...")

        # Load feature information
        self._feature_info = FeatureInfoDto.load(os.path.join(self.data_dir, features_filename))

        # Load NCF model
        self._model = NCFModel.load_model(os.path.join(self.models_dir, model_filename))

        logger.info(f"NCF model and features loaded successfully on device: {self.device}")
        return self._model, self._feature_info
