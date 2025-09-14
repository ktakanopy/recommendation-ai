# domain/ports/repositories/feature_encoder_repository.py
from abc import ABC, abstractmethod
from typing import Tuple

from sklearn.preprocessing import LabelEncoder, OneHotEncoder


class FeatureEncoderRepository(ABC):
    @property
    @abstractmethod
    def gender_encoder(self) -> LabelEncoder:
        pass

    @property
    @abstractmethod
    def age_encoder(self) -> OneHotEncoder:
        pass

    @property
    @abstractmethod
    def occupation_encoder(self) -> OneHotEncoder:
        pass

    def load_encoders(self) -> Tuple[LabelEncoder, OneHotEncoder, OneHotEncoder]:
        pass
