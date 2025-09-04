import os
import pickle
from typing import Tuple

import pandas as pd
import torch

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ModelInferenceManager:
    """Infrastructure service for managing NCF model inference"""

    def __init__(self, models_dir: str = "models",  device: str = "cpu", data_dir: str = "data"):
        self.models_dir = models_dir
        self.data_dir = data_dir
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self._model = None
        self._feature_info = None

    def load_movie_data(self, movie_filename: str) -> pd.DataFrame:
        """Load movie data from file"""
        movie_path = os.path.join(self.data_dir, movie_filename)
        return pd.read_csv(movie_path)
 
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
        self._feature_info = self._load_feature_info(features_filename)

        # Load NCF model
        self._model = self._load_ncf_model(model_filename)

        logger.info(f"NCF model and features loaded successfully on device: {self.device}")
        return self._model, self._feature_info

    def _load_feature_info(self, features_filename: str) -> FeatureInfoDto:
        """Load preprocessed feature information"""
        features_path = os.path.join("data/processed_data", features_filename)

        logger.info(f"Loading feature info from: {features_path}")

        try:
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

        except Exception as e:
            logger.warning(f"Error loading feature info: {str(e)}, creating dummy feature info")
            raise e

    def _load_ncf_model(self, model_filename: str) -> NCFModel:
        """Load trained NCF model"""
        model_path = os.path.join(self.models_dir, model_filename)

        if not os.path.exists(model_path):
            logger.warning(f"Model file not found: {model_path}, creating dummy NCF model for testing")
            return self._create_dummy_ncf_model()

        logger.info(f"Loading NCF model from: {model_path}")

        try:
            # Try to load using NCF model's load method
            model = NCFModel.load_model(model_path)
            model.to(self.device)
            model.eval()
            logger.info(f"Successfully loaded NCF model from {model_path}")
            return model

        except Exception as e:
            logger.warning(f"Failed to load NCF model: {str(e)}, creating dummy model for testing")
            raise e


    def get_device(self) -> torch.device:
        """Get the device being used for inference"""
        return self.device
