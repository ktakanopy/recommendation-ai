import os
import pickle
from typing import Tuple

import torch

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.domain.models.deep_learning.two_tower_model import TwoTowerModel
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ModelInferenceManager:
    """Infrastructure service for managing ML model inference"""

    def __init__(self, models_dir: str = "models", device: str = "cpu"):
        self.models_dir = models_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._feature_info = None

    def load_model_and_features(
        self,
        model_filename: str = "two_tower_model_all_features.pth",
        features_filename: str = "preprocessed_features.pkl",
    ) -> Tuple[TwoTowerModel, FeatureInfoDto]:
        """Load trained model and feature information"""

        if self._model is not None and self._feature_info is not None:
            return self._model, self._feature_info

        # Load feature information first
        self._feature_info = self._load_feature_info(features_filename)

        # Load model
        self._model = self._load_model(model_filename, self._feature_info)

        logger.info(f"Model and features loaded successfully on device: {self.device}")
        return self._model, self._feature_info

    def _load_feature_info(self, features_filename: str) -> FeatureInfoDto:
        """Load preprocessed feature information"""
        features_path = os.path.join("data/processed_data", features_filename)

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Features file not found: {features_path}")

        logger.info(f"Loading feature info from: {features_path}")

        with open(features_path, "rb") as f:
            save_data = pickle.load(f)

        additional_feature_info = save_data["additional_feature_info"]

        # Move tensor data to correct device
        if "sentence_embeddings" in additional_feature_info:
            sentence_embeddings = additional_feature_info["sentence_embeddings"]
            for key, value in sentence_embeddings.items():
                if isinstance(value, torch.Tensor):
                    sentence_embeddings[key] = value.to(self.device)

        return FeatureInfoDto.from_dict(additional_feature_info)

    def _load_model(self, model_filename: str, feature_info: FeatureInfoDto) -> TwoTowerModel:
        """Load trained model"""
        model_path = os.path.join(self.models_dir, model_filename)
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        logger.info(f"Loading model from: {model_path}")

        # Create model instance
        model = TwoTowerModel(
            layer_sizes=[32, 32, 32],  # TODO: Get from config
            unique_movie_titles=feature_info.unique_movie_titles,
            unique_user_ids=feature_info.unique_user_ids,
            embedding_size=64,  # TODO: Get from config
            additional_feature_info=feature_info.to_dict(),
            device=str(self.device),
        )

        # Load state dict
        state_dict = torch.load(model_path, map_location=self.device)

        # Filter state dict to match model
        model_dict = model.state_dict()
        filtered_state_dict = {
            k: v for k, v in state_dict.items() if k in model_dict and v.shape == model_dict[k].shape
        }

        model.load_state_dict(filtered_state_dict, strict=False)
        model.to(self.device)
        model.eval()

        return model

    def get_device(self) -> torch.device:
        """Get the device being used for inference"""
        return self.device
