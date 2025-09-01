from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class SentenceEmbeddingsDto:
    title_to_idx: Dict[str, int]
    embedding_matrix: torch.Tensor
    embedding_dim: int


@dataclass
class FeatureInfoDto:
    age_mean: float
    age_std: float
    sentence_embeddings: SentenceEmbeddingsDto
    unique_movie_titles: List[str]
    unique_user_ids: List[str]

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureInfoDto":
        sentence_embeddings_dict = data["sentence_embeddings"]
        sentence_embeddings = SentenceEmbeddingsDto(
            title_to_idx=sentence_embeddings_dict["title_to_idx"],
            embedding_matrix=sentence_embeddings_dict["embedding_matrix"],
            embedding_dim=sentence_embeddings_dict["embedding_dim"],
        )

        return cls(
            age_mean=data["age_mean"],
            age_std=data["age_std"],
            sentence_embeddings=sentence_embeddings,
            unique_movie_titles=data["unique_movie_titles"],
            unique_user_ids=data["unique_user_ids"],
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "sentence_embeddings": {
                "title_to_idx": self.sentence_embeddings.title_to_idx,
                "embedding_matrix": self.sentence_embeddings.embedding_matrix,
                "embedding_dim": self.sentence_embeddings.embedding_dim,
            },
            "unique_movie_titles": self.unique_movie_titles,
            "unique_user_ids": self.unique_user_ids,
        }
