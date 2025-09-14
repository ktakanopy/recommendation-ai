import os
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

from neural_recommendation.domain.exceptions import AnnoyIndexError, FeatureProcessingError
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class NCFFeatureProcessor:
    """Feature processor for NCF model with optimized user and movie feature processing"""

    def __init__(self, debug: bool = False):
        self.user_encoders = {}
        self.movie_encoders = {}
        self.sentence_model: Optional[SentenceTransformer] = None
        self.user_features_cache: Dict[int, torch.Tensor] = {}
        self.movie_features_cache: Dict[int, torch.Tensor] = {}
        self.debug = debug

        # sklearn encoders for user features
        self.gender_encoder = LabelEncoder()
        self.age_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.occupation_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
        self.encoders_fitted = False

        # Feature dimensions
        self.user_feature_dim: Optional[int] = None
        self.movie_feature_dim: Optional[int] = None
        self.movie_annoy_index: Optional[AnnoyIndex] = None
        self.movie_annoy_metric: Optional[str] = None
        self.user_annoy_index: Optional[AnnoyIndex] = None
        self.user_annoy_metric: Optional[str] = None

    def prepare_user_features(self, users_df: pd.DataFrame) -> pd.DataFrame:
        """Prepare user features: gender, age, occupation one-hot encoding"""
        if self.debug:
            logger.info("Preparing user features...")

        # Fit and transform using sklearn encoders
        if self.debug:
            logger.info("Fitting sklearn encoders...")

        # Gender encoding (M=1, F=0) using LabelEncoder
        try:
            gender_encoded = self.gender_encoder.fit_transform(users_df["gender"].values).astype(float)
        except Exception as e:
            raise FeatureProcessingError("Failed to encode gender") from e

        # Age one-hot encoding using OneHotEncoder
        try:
            age_onehot = self.age_encoder.fit_transform(users_df["age"].values.reshape(-1, 1))
        except Exception as e:
            raise FeatureProcessingError("Failed to encode age") from e

        # Occupation one-hot encoding using OneHotEncoder
        try:
            occupation_onehot = self.occupation_encoder.fit_transform(users_df["occupation"].values.reshape(-1, 1))
        except Exception as e:
            raise FeatureProcessingError("Failed to encode occupation") from e

        # Mark encoders as fitted
        self.encoders_fitted = True

        if self.debug:
            logger.info(f"Gender categories: {self.gender_encoder.classes_}")
            logger.info(f"Age categories: {self.age_encoder.categories_[0]}")
            logger.info(f"Occupation categories: {self.occupation_encoder.categories_[0]}")

        # Convert to DataFrames for consistency with original format
        age_onehot_df = pd.DataFrame(age_onehot, columns=[f"age_{cat}" for cat in self.age_encoder.categories_[0]])
        occupation_onehot_df = pd.DataFrame(
            occupation_onehot, columns=[f"occ_{cat}" for cat in self.occupation_encoder.categories_[0]]
        )

        # Combine all user features into a single DataFrame
        feature_columns = ["user_id"]
        feature_data = [users_df["user_id"].values]

        # Add gender
        feature_columns.append("gender")
        feature_data.append(gender_encoded)

        # Add age features
        for col in age_onehot_df.columns:
            feature_columns.append(col)
            feature_data.append(age_onehot_df[col].values)

        # Add occupation features
        for col in occupation_onehot_df.columns:
            feature_columns.append(col)
            feature_data.append(occupation_onehot_df[col].values)

        # Create feature matrix
        feature_matrix = np.column_stack(feature_data)
        user_features = pd.DataFrame(feature_matrix, columns=feature_columns)

        # Ensure all feature columns (except user_id) are float
        for col in user_features.columns:
            if col != "user_id":
                user_features[col] = user_features[col].astype(float)

        if self.debug:
            logger.info(f"User features shape: {user_features.shape}")
            logger.info(f"User feature columns: {list(user_features.columns)}")

        # Cache features for quick lookup (keep on CPU)
        for _, row in user_features.iterrows():
            user_id = int(row["user_id"])
            # Get feature values excluding user_id and convert to numpy array
            feature_values = row.drop("user_id").values.astype(np.float32)
            features = torch.tensor(feature_values, dtype=torch.float32)
            self.user_features_cache[user_id] = features

        self.user_feature_dim = len(user_features.columns) - 1  # Exclude user_id
        logger.info(f"User feature dimension: {self.user_feature_dim}")
        return user_features

    def _prepare_embeddings(self, movies_df: pd.DataFrame, device: str = "cpu") -> torch.Tensor:
        """Prepare movie features using sentence transformers for title and genres"""
        if self.debug:
            logger.info("Preparing movie features with sentence transformers...")

        # Initialize sentence transformer
        if self.sentence_model is None:
            if self.debug:
                logger.info("Loading sentence transformer model...")
            self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2", device=device)

        # Encode movie titles
        logger.info("Encoding movie titles...")
        try:
            titles = movies_df["title"].tolist()
            title_embeddings = self.sentence_model.encode(titles, convert_to_tensor=True, device=device, batch_size=64)
        except Exception as e:
            raise FeatureProcessingError("Failed to encode titles") from e

        # Process genres and encode them
        logger.info("Encoding movie genres...")
        genre_texts = []
        for genres_str in movies_df["genres"]:
            # Convert pipe-separated genres to readable text
            genres_list = genres_str.split("|")
            genre_text = " ".join(genres_list).replace("Children's", "Children")
            genre_texts.append(genre_text)

        try:
            genre_embeddings = self.sentence_model.encode(
                genre_texts, convert_to_tensor=True, device=device, batch_size=64
            )
        except Exception as e:
            raise FeatureProcessingError("Failed to encode genres") from e

        # Concatenate title and genre embeddings
        try:
            movie_embeddings = torch.cat([title_embeddings, genre_embeddings], dim=1)
        except Exception as e:
            raise FeatureProcessingError("Failed to concatenate embeddings") from e

        if self.debug:
            logger.info(f"Title embeddings shape: {title_embeddings.shape}")
            logger.info(f"Genre embeddings shape: {genre_embeddings.shape}")
            logger.info(f"Combined movie embeddings shape: {movie_embeddings.shape}")

        # Normalize the embeddings for better training stability
        logger.info("Normalizing movie embeddings...")
        try:
            movie_embeddings_normalized = F.normalize(movie_embeddings, p=2, dim=1)
        except Exception as e:
            raise FeatureProcessingError("Failed to normalize embeddings") from e

        # Print normalization stats
        if self.debug:
            logger.info(
                f"Original embeddings - mean: {movie_embeddings.mean().item():.4f}, std: {movie_embeddings.std().item():.4f}"
            )
            logger.info(
                f"Normalized embeddings - mean: {movie_embeddings_normalized.mean().item():.4f}, std: {movie_embeddings_normalized.std().item():.4f}"
            )

        # Cache features for quick lookup (move to CPU for storage)
        for idx, row in movies_df.iterrows():
            movie_id = int(row["movie_id"])
            # Store normalized embeddings on CPU to avoid memory issues
            self.movie_features_cache[movie_id] = movie_embeddings_normalized[idx].cpu()

        self.movie_feature_dim = movie_embeddings_normalized.shape[1]
        if self.debug:
            logger.info(f"Movie feature dimension: {self.movie_feature_dim}")
        return movie_embeddings_normalized

    def prepare_movie_features(self, movies_df: pd.DataFrame, device: str = "cpu") -> torch.Tensor:
        movie_embeddings_normalized = self._prepare_embeddings(movies_df, device=device)

        title_to_idx = movies_df.set_index("title")["movie_id"].to_dict()
        idx_to_title = movies_df.set_index("movie_id")["title"].to_dict()

        movies_genres_dict = movies_df.set_index("movie_id")["genres"].to_dict()

        self.movie_info_features_cache = {
            "movie_embeddings": movie_embeddings_normalized,
            "title_to_idx": title_to_idx,
            "idx_to_title": idx_to_title,
            "movies_genres_dict": movies_genres_dict,
        }
        return self.movie_info_features_cache

    def build_movie_annoy(self, metric: str = "angular", n_trees: int = 50) -> None:
        if self.movie_feature_dim is None:
            raise AnnoyIndexError("Movie features not prepared. Call prepare_movie_features first.")
        if not self.movie_features_cache:
            raise AnnoyIndexError("Movie features cache is empty.")
        index = AnnoyIndex(self.movie_feature_dim, metric)
        for movie_id, features in self.movie_features_cache.items():
            index.add_item(int(movie_id), features.tolist())
        index.build(n_trees)
        self.movie_annoy_index = index
        self.movie_annoy_metric = metric

    def save_movie_annoy(self, dir_path: str, index_path: str, mapping_path: Optional[str] = None) -> None:
        if self.movie_annoy_index is None:
            raise AnnoyIndexError("Annoy index not built. Call build_movie_annoy first.")
        full_index_path = os.path.join(dir_path, index_path)
        print(f"Saving movie annoy index to {full_index_path}")
        self.movie_annoy_index.save(full_index_path)
        mp = mapping_path or f"{index_path}.meta.pkl"
        full_mapping_path = os.path.join(dir_path, mp)
        with open(full_mapping_path, "wb") as f:
            pickle.dump(
                {
                    "movie_feature_dim": self.movie_feature_dim,
                    "metric": self.movie_annoy_metric,
                },
                f,
            )

    def load_movie_annoy(
        self, dir_path: str, index_path: str, mapping_path: Optional[str] = None
    ) -> "NCFFeatureProcessor":
        mp = mapping_path or f"{index_path}.meta.pkl"
        full_mapping_path = os.path.join(dir_path, mp)
        try:
            with open(full_mapping_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise AnnoyIndexError("Failed to load movie Annoy meta") from e
        dim = data["movie_feature_dim"]
        metric = data["metric"]
        index = AnnoyIndex(dim, metric)
        full_index_path = os.path.join(dir_path, index_path)
        try:
            index.load(full_index_path)
        except Exception as e:
            raise AnnoyIndexError("Failed to load movie Annoy index") from e
        self.movie_annoy_index = index
        self.movie_annoy_metric = metric
        self.movie_feature_dim = dim
        return self

    def get_similar_movies(self, movie_id: int, top_k: int = 10) -> List[int]:
        if self.movie_annoy_index is None:
            raise AnnoyIndexError("Annoy index not built. Call build_movie_annoy or load_movie_annoy first.")
        neighbors = self.movie_annoy_index.get_nns_by_item(int(movie_id), top_k + 1)
        result: List[int] = []
        for n in neighbors:
            if n != movie_id:
                result.append(int(n))
            if len(result) >= top_k:
                break
        return result

    def build_user_annoy(self, metric: str = "angular", n_trees: int = 50) -> None:
        if self.user_feature_dim is None:
            raise AnnoyIndexError("User features not prepared. Call prepare_user_features first.")
        if not self.user_features_cache:
            raise AnnoyIndexError("User features cache is empty.")
        index = AnnoyIndex(self.user_feature_dim, metric)
        for user_id, features in self.user_features_cache.items():
            index.add_item(int(user_id), features.tolist())
        index.build(n_trees)
        self.user_annoy_index = index
        self.user_annoy_metric = metric

    def save_user_annoy(self, dir_path: str, index_path: str, mapping_path: Optional[str] = None) -> None:
        if self.user_annoy_index is None:
            raise AnnoyIndexError("Annoy index not built. Call build_user_annoy first.")
        full_index_path = os.path.join(dir_path, index_path)
        print(f"Saving user annoy index to {full_index_path}")
        self.user_annoy_index.save(full_index_path)
        mp = mapping_path or f"{index_path}.meta.pkl"
        full_mapping_path = os.path.join(dir_path, mp)
        with open(full_mapping_path, "wb") as f:
            pickle.dump(
                {
                    "user_feature_dim": self.user_feature_dim,
                    "metric": self.user_annoy_metric,
                },
                f,
            )

    def load_user_annoy(
        self, dir_path: str, index_path: str, mapping_path: Optional[str] = None
    ) -> "NCFFeatureProcessor":
        mp = mapping_path or f"{index_path}.meta.pkl"
        full_mapping_path = os.path.join(dir_path, mp)
        try:
            with open(full_mapping_path, "rb") as f:
                data = pickle.load(f)
        except Exception as e:
            raise AnnoyIndexError("Failed to load user Annoy meta") from e
        dim = data["user_feature_dim"]
        metric = data["metric"]
        index = AnnoyIndex(dim, metric)
        full_index_path = os.path.join(dir_path, index_path)
        try:
            index.load(full_index_path)
        except Exception as e:
            raise AnnoyIndexError("Failed to load user Annoy index") from e
        self.user_annoy_index = index
        self.user_annoy_metric = metric
        self.user_feature_dim = dim
        return self

    def get_similar_users(self, user_id: int, top_k: int = 10) -> List[int]:
        if self.user_annoy_index is None:
            raise AnnoyIndexError("Annoy index not built. Call build_user_annoy or load_user_annoy first.")
        neighbors = self.user_annoy_index.get_nns_by_item(int(user_id), top_k + 1)
        result: List[int] = []
        for n in neighbors:
            if n != user_id:
                result.append(int(n))
            if len(result) >= top_k:
                break
        return result

    def get_user_features(self, user_id: int) -> torch.Tensor:
        """Get cached user features"""
        if self.user_feature_dim is None:
            raise ValueError("User features not prepared. Call prepare_user_features first.")

        if user_id in self.user_features_cache:
            return self.user_features_cache[user_id]
        else:
            raise ValueError(f"User {user_id} not found in user features cache")

    def save_movie_info_features_cache(self, dir_path: str, path: str) -> None:
        joined_path = os.path.join(dir_path, path)
        with open(joined_path, "wb") as f:
            pickle.dump(self.movie_info_features_cache, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_movie_features(self, movie_id: int) -> torch.Tensor:
        """Get cached movie features"""
        if self.movie_feature_dim is None:
            raise ValueError("Movie features not prepared. Call prepare_movie_features first.")
        if movie_id in self.movie_features_cache:
            return self.movie_features_cache[movie_id]
        else:
            raise ValueError(f"Movie {movie_id} not found in movie features cache")

    def process_user_demographics(self, user_demographics: Dict[str, Any]) -> torch.Tensor:
        """
        Process user demographics into feature vector format using fitted sklearn encoders

        Args:
            user_demographics: dict with keys: 'gender', 'age', 'occupation'
                - gender: 'M' or 'F'
                - age: int (age category from training data)
                - occupation: int (occupation code from training data)

        Returns:
            torch.Tensor: User feature vector (same format as cached features)
        """
        if not self.encoders_fitted:
            raise ValueError("Encoders not fitted. Call prepare_user_features first.")

        try:
            # Gender encoding using fitted LabelEncoder
            gender_encoded = self.gender_encoder.transform([user_demographics["gender"]])[0].astype(float)

            # Age one-hot encoding using fitted OneHotEncoder
            age_onehot = self.age_encoder.transform([[user_demographics["age"]]])[0]

            # Occupation one-hot encoding using fitted OneHotEncoder
            occupation_onehot = self.occupation_encoder.transform([[user_demographics["occupation"]]])[0]

            # Combine all features (same order as prepare_user_features)
            feature_vector = np.concatenate([[gender_encoded], age_onehot, occupation_onehot])

            return torch.tensor(feature_vector, dtype=torch.float32)

        except Exception as e:
            logger.error(f"Error processing user demographics: {str(e)}")
            # Return zero vector as fallback
            if self.user_feature_dim is not None:
                return torch.zeros(self.user_feature_dim)
            raise

    def get_encoder_info(self) -> Dict[str, Any]:
        """
        Get information about the fitted encoders

        Returns:
            dict: Information about encoders and their categories
        """
        if not self.encoders_fitted:
            return {"error": "Encoders not fitted yet"}

        return {
            "gender_classes": self.gender_encoder.classes_.tolist(),
            "age_categories": self.age_encoder.categories_[0].tolist(),
            "occupation_categories": self.occupation_encoder.categories_[0].tolist(),
            "total_features": 1 + len(self.age_encoder.categories_[0]) + len(self.occupation_encoder.categories_[0]),
            "user_feature_dim": self.user_feature_dim,
            "movie_feature_dim": self.movie_feature_dim,
        }

    def save_encoders(self, dir_path: str, path: str) -> None:
        joined_path = os.path.join(dir_path, path)
        with open(joined_path, "wb") as f:
            encoders = {
                "gender_encoder": self.gender_encoder,
                "age_encoder": self.age_encoder,
                "occupation_encoder": self.occupation_encoder,
            }
            pickle.dump(encoders, f, protocol=pickle.HIGHEST_PROTOCOL)

    def get_batch_user_features(self, user_ids: List[int], device: str = "cpu") -> torch.Tensor:
        """Get batched user features for multiple users"""
        batch_features = []
        for user_id in user_ids:
            features = self.get_user_features(user_id)
            batch_features.append(features)

        return torch.stack(batch_features).to(device)

    def get_batch_movie_features(self, movie_ids: List[int], device: str = "cpu") -> torch.Tensor:
        """Get batched movie features for multiple movies"""
        batch_features = []
        for movie_id in movie_ids:
            features = self.get_movie_features(movie_id)
            batch_features.append(features)

        return torch.stack(batch_features).to(device)

    def save_movie_features_cache(self, dir_path: str, path: str) -> None:
        joined_path = os.path.join(dir_path, path)
        with open(joined_path, "wb") as f:
            pickle.dump(self.movie_features_cache, f, protocol=pickle.HIGHEST_PROTOCOL)
