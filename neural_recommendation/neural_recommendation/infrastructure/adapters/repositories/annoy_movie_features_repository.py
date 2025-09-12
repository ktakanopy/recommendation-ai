from typing import Dict, List
from neural_recommendation.domain.ports.repositories.movie_features_repository import MovieFeaturesRepository
from annoy import AnnoyIndex
import torch
import os
import pickle
from neural_recommendation.domain.exceptions import RepositoryError, NotFoundError, AnnoyIndexError


class AnnoyMovieFeaturesRepository(MovieFeaturesRepository):
    def __init__(self, data_path: str, index_path: str, top_popular_movies_path: str, movie_features_cache_path: str):
        self.index_path = index_path
        self.data_path = data_path
        self.top_popular_movies_path = top_popular_movies_path
        self.movie_features_cache_path = movie_features_cache_path

        self.annoy_index = self.load()
        self.top_popular_movies = self.load_top_popular_movies()
        self.movie_features_cache = self.load_movie_features_cache()

    def get_features(self, movie_id: int) -> torch.Tensor:
        try:
            return self.movie_features_cache[movie_id]
        except KeyError as e:
            raise NotFoundError(f"Movie features not found for id={movie_id}") from e

    def get_similar_movies(self, movie_id: int, top_k: int):
        if self.annoy_index is None:
            raise AnnoyIndexError("Annoy index not loaded")
        try:
            return self.annoy_index.get_nns_by_item(movie_id, top_k)
        except Exception as e:
            raise RepositoryError("Failed to query Annoy index for similar movies") from e

    def load_top_popular_movies(self) -> List[int]:
        try:
            with open(os.path.join(self.data_path, self.top_popular_movies_path), "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RepositoryError("Failed to load top popular movies") from e

    def get_top_popular_movies(self, top_k: int) -> List[int]:
        return self.top_popular_movies[:top_k]

    def load_movie_features_cache(self) -> Dict[int, torch.Tensor]:
        try:
            with open(os.path.join(self.data_path, self.movie_features_cache_path), "rb") as f:
                return pickle.load(f)
        except Exception as e:
            raise RepositoryError("Failed to load movie features cache") from e

    def load(self):
        try:
            mp = f"{self.index_path}.meta.pkl"
            full_mapping_path = os.path.join(self.data_path, mp)
            with open(full_mapping_path, "rb") as f:
                data = pickle.load(f)

            dim = data["movie_feature_dim"]
            metric = data["metric"]

            index = AnnoyIndex(dim, metric)
            full_index_path = os.path.join(self.data_path, self.index_path)
            index.load(full_index_path)

            return index
        except Exception as e:
            raise RepositoryError("Failed to load Annoy movie index") from e
