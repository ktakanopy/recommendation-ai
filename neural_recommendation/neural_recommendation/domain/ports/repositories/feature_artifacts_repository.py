from abc import ABC, abstractmethod

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto


class FeatureArtifactsRepository(ABC):
    @abstractmethod
    def load_feature_info(self) -> FeatureInfoDto:
        pass

    @abstractmethod
    def save_feature_info(self, feature_info: FeatureInfoDto):
        pass
