from typing import Any, Dict, List, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from neural_recommendation.domain.models.deep_learning.movie_model import MovieModel
from neural_recommendation.domain.models.deep_learning.user_model import UserModel


class TwoTowerModel(nn.Module):
    def __init__(
        self,
        layer_sizes: List[int],
        unique_user_ids: List[str],
        embedding_size: int,
        additional_feature_info: Dict,
        device: str = "cpu",
        dropout_rate: float = 0.3,
    ):
        super().__init__()
        self.device = device
        self.dropout_rate = dropout_rate
        self.user_model = UserModel(
            unique_user_ids,
            embedding_size,
            additional_feature_info,
            device,
        )
        self.movie_model = MovieModel(
            additional_feature_info,
            embedding_size,
            device,
        )
        user_input_size = self._calculate_user_input_size(embedding_size)
        movie_input_size = self._calculate_movie_input_size(embedding_size, additional_feature_info)
        self.query_tower = self._build_tower([user_input_size] + layer_sizes)
        self.candidate_tower = self._build_tower([movie_input_size] + layer_sizes)

    def _calculate_user_input_size(self, embedding_size: int) -> int:
        return embedding_size

    def _calculate_movie_input_size(
        self,
        embedding_size: int,
        additional_feature_info: Dict,
    ) -> int:
        return embedding_size

    def _build_tower(self, layer_sizes: List[int]) -> nn.Module:
        layers = []
        for i in range(len(layer_sizes) - 1):
            layers.append(nn.Linear(layer_sizes[i], layer_sizes[i + 1]))
            if i < len(layer_sizes) - 2:
                layers.append(nn.LayerNorm(layer_sizes[i + 1]))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(self.dropout_rate))
        return nn.Sequential(*layers)

    def forward(self, user_inputs: Dict[str, Any], movie_inputs: Dict[str, Any]) -> Tuple[torch.Tensor, torch.Tensor]:
        user_embeddings = self.user_model(user_inputs)
        movie_embeddings = self.movie_model(movie_inputs)
        query_embeddings = self.query_tower(user_embeddings)
        candidate_embeddings = self.candidate_tower(movie_embeddings)
        return query_embeddings, candidate_embeddings

    def compute_loss(
        self,
        user_inputs: Dict[str, Any],
        movie_inputs: Dict[str, Any],
        temperature: float = 0.1,
    ) -> torch.Tensor:
        query_embeddings, candidate_embeddings = self.forward(user_inputs, movie_inputs)
        query_embeddings = F.normalize(query_embeddings, p=2, dim=1)
        candidate_embeddings = F.normalize(candidate_embeddings, p=2, dim=1)
        logits = torch.matmul(query_embeddings, candidate_embeddings.T) / temperature
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_loss_with_negatives(
        self,
        user_inputs: Dict[str, Any],
        positive_movie_inputs: Dict[str, Any],
        negative_movie_inputs: Dict[str, Any],
    ) -> torch.Tensor:
        """
        Compute loss using explicit negative sampling.

        Args:
            user_inputs: User features [batch_size, ...]
            positive_movie_inputs: Positive movie features [batch_size, ...]
            negative_movie_inputs: Negative movie features [batch_size, num_negatives, ...]
        """
        batch_size = len(user_inputs["user_id"])
        num_negatives = negative_movie_inputs["movie_idx"].size(1)

        # Get user embeddings
        user_embeddings = self.user_model(user_inputs)  # [batch_size, embed_dim]
        user_embeddings = F.normalize(user_embeddings, p=2, dim=1)

        # Get positive movie embeddings
        positive_embeddings = self.movie_model(positive_movie_inputs)  # [batch_size, embed_dim]
        positive_embeddings = F.normalize(positive_embeddings, p=2, dim=1)

        # Get negative movie embeddings
        # Reshape negative inputs to process all negatives at once
        neg_movie_indices = negative_movie_inputs["movie_idx"].view(-1)  # [batch_size * num_negatives]
        neg_inputs_flat = {"movie_idx": neg_movie_indices}
        negative_embeddings = self.movie_model(neg_inputs_flat)  # [batch_size * num_negatives, embed_dim]
        negative_embeddings = F.normalize(negative_embeddings, p=2, dim=1)
        negative_embeddings = negative_embeddings.view(
            batch_size, num_negatives, -1
        )  # [batch_size, num_negatives, embed_dim]

        # Apply towers
        user_query = self.query_tower(user_embeddings)  # [batch_size, final_dim]
        positive_candidate = self.candidate_tower(positive_embeddings)  # [batch_size, final_dim]

        # Process negatives through candidate tower
        negative_embeddings_flat = negative_embeddings.view(-1, negative_embeddings.size(-1))
        negative_candidates = self.candidate_tower(negative_embeddings_flat)  # [batch_size * num_negatives, final_dim]
        negative_candidates = negative_candidates.view(
            batch_size, num_negatives, -1
        )  # [batch_size, num_negatives, final_dim]

        # Normalize final embeddings
        user_query = F.normalize(user_query, p=2, dim=1)
        positive_candidate = F.normalize(positive_candidate, p=2, dim=1)
        negative_candidates = F.normalize(negative_candidates, p=2, dim=2)

        # Compute similarities
        positive_scores = torch.sum(user_query * positive_candidate, dim=1)  # [batch_size]
        negative_scores = torch.bmm(
            user_query.unsqueeze(1),  # [batch_size, 1, final_dim]
            negative_candidates.transpose(1, 2),  # [batch_size, final_dim, num_negatives]
        ).squeeze(1)  # [batch_size, num_negatives]

        # Binary classification loss
        positive_loss = F.binary_cross_entropy_with_logits(positive_scores, torch.ones_like(positive_scores))
        negative_loss = F.binary_cross_entropy_with_logits(negative_scores, torch.zeros_like(negative_scores))

        return positive_loss + negative_loss

    def compute_sampled_softmax_loss(
        self,
        user_inputs: Dict[str, Any],
        positive_movie_inputs: Dict[str, Any],
        num_sampled_negatives: int = 1000,
    ) -> torch.Tensor:
        """
        Compute loss using sampled softmax with randomly sampled negative movies.

        Args:
            user_inputs: User features [batch_size, ...]
            positive_movie_inputs: Positive movie features [batch_size, ...]
            num_sampled_negatives: Number of negative movies to sample for softmax
        """
        batch_size = len(user_inputs["user_id"])
        device = user_inputs["user_id"].device

        # Get user embeddings
        user_embeddings = self.user_model(user_inputs)
        user_query = self.query_tower(user_embeddings)
        user_query = F.normalize(user_query, p=2, dim=1)

        # Get positive movie embeddings
        positive_embeddings = self.movie_model(positive_movie_inputs)
        positive_candidate = self.candidate_tower(positive_embeddings)
        positive_candidate = F.normalize(positive_candidate, p=2, dim=1)

        # Sample negative movies
        total_movies = len(self.movie_model.title_to_idx)
        sampled_movie_indices = torch.randint(0, total_movies, (num_sampled_negatives,), device=device)
        sampled_movie_inputs = {"movie_idx": sampled_movie_indices}
        sampled_embeddings = self.movie_model(sampled_movie_inputs)
        sampled_candidates = self.candidate_tower(sampled_embeddings)
        sampled_candidates = F.normalize(sampled_candidates, p=2, dim=1)

        # Compute similarities
        positive_scores = torch.sum(user_query * positive_candidate, dim=1)  # [batch_size]
        negative_scores = torch.matmul(user_query, sampled_candidates.T)  # [batch_size, num_sampled]

        # Combine positive and negative scores
        all_scores = torch.cat(
            [
                positive_scores.unsqueeze(1),  # [batch_size, 1]
                negative_scores,  # [batch_size, num_sampled]
            ],
            dim=1,
        )  # [batch_size, 1 + num_sampled]

        # Labels: positive is always at index 0
        labels = torch.zeros(batch_size, dtype=torch.long, device=device)

        return F.cross_entropy(all_scores, labels)
