import os
import pickle
from typing import Tuple

from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from neural_recommendation.domain.ports.repositories.feature_encoder_repository import (
    FeatureEncoderRepository,
)
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class PickleFeatureEncoderRepository(FeatureEncoderRepository):
    def __init__(self, data_path: str, encoder_path: str):
        self.data_path = data_path
        self.index_path = encoder_path
        self._gender_encoder, self._age_encoder, self._occupation_encoder = self._load_encoders(data_path, encoder_path)

    @property
    def gender_encoder(self) -> LabelEncoder:
        return self._gender_encoder

    @property
    def age_encoder(self) -> OneHotEncoder:
        return self._age_encoder

    @property
    def occupation_encoder(self) -> OneHotEncoder:
        return self._occupation_encoder

    def _load_encoders(self, dir_path: str, path: str) -> Tuple[LabelEncoder, OneHotEncoder, OneHotEncoder]:
        joined_path = os.path.join(dir_path, path)
        with open(joined_path, "rb") as f:
            encoders = pickle.load(f)
        return encoders["gender_encoder"], encoders["age_encoder"], encoders["occupation_encoder"]
