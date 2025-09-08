import pickle
from dataclasses import dataclass
from typing import Any, Dict, List

import torch


@dataclass
class SentenceEmbeddingsDto:
    embedding_matrix: torch.Tensor
    embedding_dim: int
    title_to_idx: Dict[str, int]
    idx_to_title: Dict[int, str]
    movies_genres_dict: Dict[str, List[str]]


@dataclass
class FeatureInfoDto:
    age_mean: float
    age_std: float
    sentence_embeddings: SentenceEmbeddingsDto

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FeatureInfoDto":
        sentence_embeddings_dict = data["sentence_embeddings"]
        sentence_embeddings = SentenceEmbeddingsDto(
            embedding_matrix=sentence_embeddings_dict["embedding_matrix"],
            embedding_dim=sentence_embeddings_dict["embedding_dim"],
            title_to_idx=sentence_embeddings_dict["title_to_idx"],
            idx_to_title=sentence_embeddings_dict["idx_to_title"],
            movies_genres_dict=sentence_embeddings_dict["movies_genres_dict"],
        )

        return cls(
            age_mean=data["age_mean"],
            age_std=data["age_std"],
            sentence_embeddings=sentence_embeddings,
        )

    def to_dict(self) -> Dict[str, Any]:
        return {
            "age_mean": self.age_mean,
            "age_std": self.age_std,
            "sentence_embeddings": {
                "embedding_matrix": self.sentence_embeddings.embedding_matrix,
                "embedding_dim": self.sentence_embeddings.embedding_dim,
                "title_to_idx": self.sentence_embeddings.title_to_idx,
                "idx_to_title": self.sentence_embeddings.idx_to_title,
                "movies_genres_dict": self.sentence_embeddings.movies_genres_dict,
            },
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self.to_dict(), f)

    @classmethod
    def load(cls, path: str):
        with open(path, "rb") as f:
            loaded = pickle.load(f)
            return cls.from_dict(loaded)
