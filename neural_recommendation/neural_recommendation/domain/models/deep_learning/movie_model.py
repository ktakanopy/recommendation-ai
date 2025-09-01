from typing import Any, Dict

import torch
from torch import nn

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class MovieModel(nn.Module):
    def __init__(
        self,
        additional_feature_info: FeatureInfoDto,
        embedding_size: int = 32,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.embedding_size = embedding_size
        self.title_to_idx = additional_feature_info.sentence_embeddings.title_to_idx
        self.sentence_embeddings = additional_feature_info.sentence_embeddings
        embedding_matrix = additional_feature_info.sentence_embeddings.embedding_matrix
        self.embedding_matrix = nn.Parameter(embedding_matrix.clone(), requires_grad=True)
        self.precomputed_embedding_dim = self.sentence_embeddings.embedding_dim
        self.projection = nn.Sequential(
            nn.Linear(self.precomputed_embedding_dim, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
            nn.ReLU(),
            nn.Linear(self.embedding_size, self.embedding_size),
            nn.LayerNorm(self.embedding_size),
        )
        logger.info(f"Using precomputed movie embeddings (dim: {self.precomputed_embedding_dim})")

    def forward(self, inputs: Dict[str, Any]) -> torch.Tensor:
        movie_indices = inputs["movie_idx"]
        movie_embeds = self.embedding_matrix[movie_indices]
        projected = self.projection(movie_embeds)
        return projected
