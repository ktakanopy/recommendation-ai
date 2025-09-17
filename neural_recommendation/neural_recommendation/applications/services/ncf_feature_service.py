from typing import Any, Dict, Optional

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

from neural_recommendation.domain.exceptions import FeatureProcessingError
from neural_recommendation.domain.ports.repositories.feature_encoder_repository import FeatureEncoderRepository
from neural_recommendation.domain.ports.services.logger import LoggerPort


class NCFFeatureService:
    """Feature processor for NCF model with optimized user and movie feature processing"""

    def __init__(self, feature_encoder_repository: FeatureEncoderRepository, logger: LoggerPort):
        self.user_encoders = {}
        self.movie_encoders = {}
        self.sentence_model: Optional[SentenceTransformer] = None
        self.feature_encoder_repository = feature_encoder_repository
        self.logger = logger

        self.gender_encoder = self.feature_encoder_repository.gender_encoder
        self.age_encoder = self.feature_encoder_repository.age_encoder
        self.occupation_encoder = self.feature_encoder_repository.occupation_encoder

    def process_user_demographics(self, user_demographics: Dict[str, Any]) -> torch.Tensor:
        """
        Process user demographics into feature vector format using fitted sklearn encoders

        Args:
            user_demographics: dict with keys: 'gender', 'age', 'occupation'
                - gender: 'M' or 'F'
                - age: int (age category from training data)
                - occupation: int (occupation code from training data)

        Returns:
            torch.Tensor: User feature vector (same format as cached features)
        """
        try:
            self.logger.info("Processing user demographics: %s", user_demographics)
            # Gender encoding using fitted LabelEncoder
            gender_encoded = self.gender_encoder.transform([user_demographics["gender"]])[0].astype(float)

            # Age one-hot encoding using fitted OneHotEncoder
            age_onehot = self.age_encoder.transform([[user_demographics["age"]]])[0]

            # Occupation one-hot encoding using fitted OneHotEncoder
            occupation_onehot = self.occupation_encoder.transform([[user_demographics["occupation"]]])[0]

            # Combine all features (same order as prepare_user_features)
            feature_vector = np.concatenate([[gender_encoded], age_onehot, occupation_onehot])

            return feature_vector

        except Exception as e:
            self.logger.error(f"Error processing user demographics: {str(e)} for user demographics: {user_demographics}")
            raise FeatureProcessingError(f"Failed to process user demographics: {str(e)} for user demographics: {user_demographics}")