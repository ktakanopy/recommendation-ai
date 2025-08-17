import pickle
from pathlib import Path

from two_towers.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto


class FeatureArtifactsPickleRepository:
    """Loads pre-computed feature artifacts"""

    def __init__(self, artifacts_path: str):
        self.artifacts_path = Path(artifacts_path)

    def load_feature_info(self) -> FeatureInfoDto:
        """Load pre-computed feature mappings and statistics"""
        with open(self.artifacts_path / "feature_info.pkl", "rb") as f:
            feature_info_dict = pickle.load(f)
        return FeatureInfoDto.from_dict(feature_info_dict)

    def save_feature_info(self, feature_info: FeatureInfoDto):
        """Save feature artifacts after training"""
        self.artifacts_path.mkdir(parents=True, exist_ok=True)
        with open(self.artifacts_path / "feature_info.pkl", "wb") as f:
            pickle.dump(feature_info.to_dict(), f)
