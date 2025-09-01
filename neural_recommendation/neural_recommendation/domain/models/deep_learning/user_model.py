from typing import Any, Dict, List, Optional

import torch
from torch import nn

from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class UserModel(nn.Module):
    def __init__(
        self,
        unique_user_ids: List[str],
        embedding_size: int = 32,
        additional_feature_info: Optional[Dict] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.additional_feature_info = additional_feature_info or {}
        self.device = device
        self.embedding_size = embedding_size
        self.user_id_to_idx = {user_id: idx for idx, user_id in enumerate(unique_user_ids)}
        self.user_embedding = nn.Embedding(len(unique_user_ids), embedding_size)
        self.additional_embeddings = nn.ModuleDict()
        self._setup_additional_embeddings(embedding_size)

        total_input_size = 4 * embedding_size  # user_id + age + gender + occupation

        self.feature_mlp = nn.Sequential(
            nn.Linear(total_input_size, embedding_size * 2),
            nn.LayerNorm(embedding_size * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_size * 2, embedding_size),
            nn.LayerNorm(embedding_size),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_size, embedding_size),
        )

        logger.info(
            f"UserModel initialized with feature MLP processing {total_input_size} -> {embedding_size} features"
        )

    def _setup_additional_embeddings(self, embedding_size: int):
        age_mean = self.additional_feature_info.age_mean
        age_std = self.additional_feature_info.age_std
        self.register_buffer("user_age_mean", torch.tensor(age_mean, dtype=torch.float32))
        self.register_buffer("user_age_std", torch.tensor(age_std, dtype=torch.float32))
        self.additional_embeddings["user_age"] = nn.Linear(1, embedding_size)

        self.additional_embeddings["gender"] = nn.Embedding(2, embedding_size)

        self.additional_embeddings["occupation"] = nn.Embedding(21, embedding_size)

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        user_indices = inputs["user_id"]
        embeddings = [self.user_embedding(user_indices)]
        embeddings.extend(self._process_additional_features(inputs))

        concatenated_features = torch.cat(embeddings, dim=1)

        processed_features = self.feature_mlp(concatenated_features)

        return processed_features

    def _process_additional_features(self, inputs: Dict[str, Any]) -> List[torch.Tensor]:
        embeddings = []

        # Process age
        ages = inputs["user_age"]
        normalized_ages = (ages - self.user_age_mean) / self.user_age_std
        embeddings.append(self.additional_embeddings["user_age"](normalized_ages.unsqueeze(-1)))

        # Process gender (0 for Female, 1 for Male)
        gender_indices = inputs["gender"]
        embeddings.append(self.additional_embeddings["gender"](gender_indices))

        # Process occupation (0-20 for different occupation types)
        occupation_indices = inputs["occupation"]
        embeddings.append(self.additional_embeddings["occupation"](occupation_indices))

        return embeddings
