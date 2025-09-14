import os
import pickle

import torch
from annoy import AnnoyIndex

from neural_recommendation.domain.exceptions import AnnoyIndexError, NotFoundError, RepositoryError
from neural_recommendation.domain.ports.repositories.user_features_repository import UserFeaturesRepository
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class AnnoyUserFeaturesRepository(UserFeaturesRepository):
    def __init__(self, data_path: str, index_path: str):
        self.index_path = index_path
        self.data_path = data_path

        self.annoy_index = self.load()

    def get_features(self, user_id: int) -> torch.Tensor:
        try:
            return self.annoy_index.get_item(user_id)
        except Exception as e:
            raise NotFoundError(f"User features not found for id={user_id}") from e

    def get_similar_users(self, user_features: torch.Tensor, top_k: int):
        if self.annoy_index is None:
            raise AnnoyIndexError("Annoy index not loaded")
        try:
            return self.annoy_index.get_nns_by_vector(user_features, top_k)
        except Exception as e:
            raise RepositoryError(f"Failed to query Annoy index for similar users: {str(e)}")

    def load(self):
        try:
            mp = f"{self.index_path}.meta.pkl"
            full_mapping_path = os.path.join(self.data_path, mp)
            with open(full_mapping_path, "rb") as f:
                data = pickle.load(f)

            dim = data["user_feature_dim"]
            metric = data["metric"]

            index = AnnoyIndex(dim, metric)
            full_index_path = os.path.join(self.data_path, self.index_path)
            index.load(full_index_path)

            return index
        except Exception as e:
            raise RepositoryError("Failed to load Annoy user index") from e
