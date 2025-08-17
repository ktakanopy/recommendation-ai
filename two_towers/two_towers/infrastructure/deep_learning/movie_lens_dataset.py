import random
from typing import Any, Dict, List

import torch
from torch.utils.data import Dataset


class MovieLensDataset(Dataset):
    def __init__(
        self,
        ratings_data: Dict[str, Any],
        user_id_to_idx: Dict[str, int],
        title_to_idx: Dict[str, int],
        num_negatives: int = 4,
        use_hard_negatives: bool = False,
        hard_negative_ratio: float = 0.5,
        include_negatives: bool = True,
    ):
        self.ratings_data = ratings_data
        self.user_id_to_idx = user_id_to_idx
        self.title_to_idx = title_to_idx
        self.num_negatives = num_negatives
        self.use_hard_negatives = use_hard_negatives
        self.hard_negative_ratio = hard_negative_ratio
        self.include_negatives = include_negatives
        self.all_movie_indices = list(range(len(title_to_idx)))

        # Create user interaction sets for negative sampling (only if needed)
        if self.include_negatives:
            self.user_interactions = self._build_user_interactions()
        else:
            self.user_interactions = {}

        # For hard negative mining (will be populated during training)
        self.movie_embeddings = None
        self.model_ref = None

    def __len__(self) -> int:
        return len(self.ratings_data["user_id"])

    def _build_user_interactions(self) -> Dict[str, set]:
        """Build a mapping of user_id to set of movie indices they've interacted with"""
        user_interactions = {}
        for i in range(len(self.ratings_data["user_id"])):
            user_id = self.ratings_data["user_id"][i]
            movie_title = self.ratings_data["movie_title"][i]
            movie_idx = self.title_to_idx.get(movie_title, 0)

            if user_id not in user_interactions:
                user_interactions[user_id] = set()
            user_interactions[user_id].add(movie_idx)

        return user_interactions

    def _sample_negatives(self, user_id: str, positive_movie_idx: int) -> List[int]:
        """Sample negative movies for a user"""
        # Get movies this user has interacted with
        user_movies = self.user_interactions.get(user_id, set())

        # Sample from movies the user hasn't interacted with
        available_movies = [idx for idx in self.all_movie_indices
                          if idx not in user_movies and idx != positive_movie_idx]

        if len(available_movies) < self.num_negatives:
            # If not enough movies, pad with random sampling with replacement
            negatives = available_movies + random.choices(
                available_movies, k=self.num_negatives - len(available_movies)
            )
        else:
            negatives = random.sample(available_movies, self.num_negatives)

        return negatives

    def set_model_for_hard_negatives(self, model, movie_embeddings):
        """Set model reference and precomputed movie embeddings for hard negative mining"""
        self.model_ref = model
        self.movie_embeddings = movie_embeddings

    def _sample_hard_negatives(self, user_id: str, positive_movie_idx: int, user_embedding=None) -> List[int]:
        """Sample hard negatives using model similarity"""
        if self.model_ref is None or self.movie_embeddings is None or user_embedding is None:
            # Fallback to random sampling
            return self._sample_negatives(user_id, positive_movie_idx)

        # Get movies this user has interacted with
        user_movies = self.user_interactions.get(user_id, set())
        available_movies = [idx for idx in self.all_movie_indices
                          if idx not in user_movies and idx != positive_movie_idx]

        if len(available_movies) < self.num_negatives:
            return available_movies + random.choices(
                available_movies, k=self.num_negatives - len(available_movies)
            )

        # Compute similarities between user and available movies
        available_embeddings = self.movie_embeddings[available_movies]  # [num_available, embed_dim]
        similarities = torch.cosine_similarity(
            user_embedding.unsqueeze(0),
            available_embeddings,
            dim=1
        )

        # Get top similar movies as hard negatives
        num_hard = min(self.num_negatives, len(available_movies))
        _, top_indices = torch.topk(similarities, num_hard)
        hard_negatives = [available_movies[i] for i in top_indices.tolist()]

        return hard_negatives

    def _sample_mixed_negatives(self, user_id: str, positive_movie_idx: int, user_embedding=None) -> List[int]:
        """Sample a mix of random and hard negatives"""
        if not self.use_hard_negatives:
            return self._sample_negatives(user_id, positive_movie_idx)

        num_hard = int(self.num_negatives * self.hard_negative_ratio)
        num_random = self.num_negatives - num_hard

        # Get hard negatives
        hard_negatives = []
        if num_hard > 0 and user_embedding is not None:
            try:
                hard_negatives = self._sample_hard_negatives(user_id, positive_movie_idx, user_embedding)[:num_hard]
            except Exception:
                # Fallback to random if hard negative mining fails
                pass

        # Get random negatives
        random_negatives = []
        if num_random > 0:
            random_negatives = self._sample_negatives(user_id, positive_movie_idx)[:num_random]

        # Combine and pad if necessary
        all_negatives = hard_negatives + random_negatives
        while len(all_negatives) < self.num_negatives:
            all_negatives.extend(self._sample_negatives(user_id, positive_movie_idx))

        return all_negatives[:self.num_negatives]

    def __getitem__(self, idx: int):
        user_id = self.ratings_data["user_id"][idx]
        user_idx = self.user_id_to_idx.get(user_id, 0)

        user_inputs = {
            "user_id": user_idx,
            "user_age": self.ratings_data["user_age"][idx],
            "gender": self.ratings_data["gender"][idx],
            "occupation": self.ratings_data["occupation"][idx],
        }

        movie_title = self.ratings_data["movie_title"][idx]
        positive_movie_idx = self.title_to_idx.get(movie_title, 0)
        positive_movie_inputs = {"movie_idx": positive_movie_idx}

        if self.include_negatives:
            # Sample negative movies (mix of random and hard negatives)
            negative_movie_indices = self._sample_mixed_negatives(user_id, positive_movie_idx)
            negative_movie_inputs = [{"movie_idx": neg_idx} for neg_idx in negative_movie_indices]
            return user_inputs, positive_movie_inputs, negative_movie_inputs
        else:
            # Return 2-tuple for evaluation
            return user_inputs, positive_movie_inputs
