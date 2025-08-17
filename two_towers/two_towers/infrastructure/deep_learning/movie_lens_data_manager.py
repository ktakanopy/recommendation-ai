import os
import pickle
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
import torch
from typing_extensions import Optional

from two_towers.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from two_towers.domain.services.feature_engineering_service import FeatureEngineeringService
from two_towers.infrastructure.deep_learning.sentence_embedding_processor import SentenceEmbeddingProcessor
from two_towers.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class DataManager:
    def __init__(
        self,
        reprocess: bool = False,
        device: str = "cpu",
        sentence_model: str = "all-MiniLM-L6-v2",
        data_dir: str = "./data/ml-1m",
        processed_data_dir: str = "./data/processed_data",
        sample_users: Optional[int] = None,
    ):
        self.data_dir = data_dir
        self.device = device
        self.sample_users = sample_users
        self.processed_data_dir = processed_data_dir
        self.sentence_processor = SentenceEmbeddingProcessor(model_name=sentence_model, device=self.device)
        self.reprocess = reprocess

    def prepare_data(self):
        cache_suffix = f"_sample_{self.sample_users}" if self.sample_users else ""
        processed_data_path = os.path.join(self.processed_data_dir, f"processed_data{cache_suffix}.pkl")
        preprocessed_features_path = os.path.join(self.processed_data_dir, f"preprocessed_features{cache_suffix}.pkl")

        if os.path.exists(preprocessed_features_path):
            try:
                return self.load_preprocessed_features()
            except Exception as e:
                logger.info(f"Error loading preprocessed features: {e}")
                logger.info("Falling back to processing from raw data...")

        logger.info("Processing from raw data...")
        self.ratings_df, self.movies_df, self.users_df = self.load_raw_data()
        self.ratings_df, self.movies_df, self.users_df = self.sample_data(
            self.ratings_df, self.movies_df, self.users_df
        )
        with open(processed_data_path, "wb") as f:
            pickle.dump(
                {
                    "ratings_df": self.ratings_df,
                    "movies_df": self.movies_df,
                    "users_df": self.users_df,
                },
                f,
            )
        return self._prepare_features()

    def _prepare_features(self) -> Tuple[Dict[str, Any], FeatureInfoDto]:
        data_df = self.ratings_df.merge(self.movies_df, on="movie_id")
        data_df = data_df.merge(self.users_df, on="user_id")
        data_df["genres_list"] = data_df["genres"].str.split("|")
        all_genres = set()
        for genres in data_df["genres_list"]:
            all_genres.update(genres)
        self.genre_to_id = {genre: idx for idx, genre in enumerate(sorted(all_genres))}
        self.id_to_genre = {idx: genre for genre, idx in self.genre_to_id.items()}
        data_df["genre_ids"] = data_df["genres_list"].apply(lambda x: [self.genre_to_id[genre] for genre in x])
        data_df["genre_names"] = data_df["genres_list"]

        logger.info("Creating genres dataframe with statistics...")
        genre_stats = []
        for genre in sorted(all_genres):
            genre_movies = data_df[data_df["genres_list"].apply(lambda x: genre in x)]

            movie_count = len(genre_movies.groupby("movie_id"))
            rating_count = len(genre_movies)
            avg_rating = genre_movies["rating"].mean() if rating_count > 0 else 0

            genre_stats.append({
                "genre": genre,
                "genre_id": self.genre_to_id[genre],
                "movie_count": movie_count,
                "rating_count": rating_count,
                "avg_rating": avg_rating,
            })

        self.genres_df = pd.DataFrame(genre_stats)
        self.genres_df = self.genres_df.sort_values("movie_count", ascending=False)
        logger.info(f"Created genres dataframe with {len(self.genres_df)} genres")

        # Use domain service for gender encoding
        data_df["gender"] = data_df["gender"].apply(FeatureEngineeringService.encode_gender)

        logger.info(
            f"Created {len([col for col in data_df.columns if col.startswith('is_') or col.endswith('_sin') or col.endswith('_cos')])} timestamp features"
        )

        self.unique_movie_titles = self.movies_df["title"].unique()
        self.unique_user_ids = [str(uid) for uid in self.users_df["user_id"].unique()]
        data_df["user_id"] = data_df["user_id"].astype(str)
        sentence_embeddings = {}
        sentence_embeddings = self.sentence_processor.precompute_embeddings(self.movies_df)

        train_size = int(0.6 * len(data_df))
        train_data = data_df.iloc[:train_size]
        age_mean = train_data["age"].mean()
        age_std = train_data["age"].std()
        logger.info(f"Age normalization stats (computed on training data only): mean={age_mean:.3f}, std={age_std:.3f}")

        additional_feature_info = {
            "age_mean": age_mean,
            "age_std": age_std,
            "sentence_embeddings": sentence_embeddings,
            "unique_movie_titles": self.unique_movie_titles,
            "unique_user_ids": self.unique_user_ids,
        }

        all_ratings = {
            "user_id": data_df["user_id"].values,
            "movie_title": data_df["title"].values,
            "user_age": data_df["age"].values,
            "gender": data_df["gender"].values,
            "occupation": data_df["occupation"].values,
        }

        feature_info_dto = FeatureInfoDto.from_dict(additional_feature_info)

        self._save_preprocessed_features(all_ratings, feature_info_dto.to_dict())

        return all_ratings, feature_info_dto

    def _save_preprocessed_features(self, all_ratings: Dict[str, Any], additional_feature_info: Dict[str, Any]):
        cache_suffix = f"_sample_{self.sample_users}" if self.sample_users else ""
        features_path = os.path.join(self.processed_data_dir, f"preprocessed_features{cache_suffix}.pkl")

        features_to_save = {}
        for key, value in additional_feature_info.items():
            if key == "sentence_embeddings":
                sentence_embeddings_save = {}
                for emb_key, emb_value in value.items():
                    if isinstance(emb_value, torch.Tensor):
                        sentence_embeddings_save[emb_key] = emb_value.cpu()
                    else:
                        sentence_embeddings_save[emb_key] = emb_value
                features_to_save[key] = sentence_embeddings_save
            elif isinstance(value, torch.Tensor):
                features_to_save[key] = value.cpu()
            else:
                features_to_save[key] = value

        save_data = {
            "all_ratings": all_ratings,
            "additional_feature_info": features_to_save,
            "ratings_df": self.ratings_df,
            "movies_df": self.movies_df,
            "users_df": self.users_df,
            "genres_df": self.genres_df,
        }

        with open(features_path, "wb") as f:
            pickle.dump(save_data, f)
        logger.info(f"Preprocessed features saved to: {features_path}")

    def load_preprocessed_features(self) -> Tuple[Dict[str, Any], FeatureInfoDto]:
        cache_suffix = f"_sample_{self.sample_users}" if self.sample_users else ""
        features_path = os.path.join(self.processed_data_dir, f"preprocessed_features{cache_suffix}.pkl")

        if not os.path.exists(features_path):
            raise FileNotFoundError(f"Preprocessed features not found: {features_path}")

        logger.info(f"Loading preprocessed features from: {features_path}")
        with open(features_path, "rb") as f:
            save_data = pickle.load(f)

        self.ratings_df = save_data["ratings_df"]
        self.movies_df = save_data["movies_df"]
        self.users_df = save_data["users_df"]

        self.genres_df = save_data.get("genres_df", pd.DataFrame())
        if not self.genres_df.empty:
            logger.info(f"Loaded genres dataframe with {len(self.genres_df)} genres")
        else:
            logger.info("No genres dataframe found")

        additional_feature_info = save_data["additional_feature_info"]
        if "sentence_embeddings" in additional_feature_info:
            sentence_embeddings = additional_feature_info["sentence_embeddings"]
            for key, value in sentence_embeddings.items():
                if isinstance(value, torch.Tensor):
                    sentence_embeddings[key] = value.to(self.device)

        all_ratings = save_data["all_ratings"]

        return all_ratings, FeatureInfoDto.from_dict(additional_feature_info)

    def load_raw_data(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        ratings_df = pd.read_csv(
            os.path.join(self.data_dir, "ratings.dat"),
            sep="::",
            names=["user_id", "movie_id", "rating", "timestamp"],
            engine="python",
        )
        movies_df = pd.read_csv(
            os.path.join(self.data_dir, "movies.dat"),
            sep="::",
            names=["movie_id", "title", "genres"],
            engine="python",
            encoding="latin-1",
        )
        users_df = pd.read_csv(
            os.path.join(self.data_dir, "users.dat"),
            sep="::",
            names=["user_id", "gender", "age", "occupation", "zip_code"],
            engine="python",
        )

        if not users_df["gender"].isin(["M", "F"]).all():
            raise ValueError("Invalid gender values found. Expected only 'M' or 'F'")

        return ratings_df, movies_df, users_df

    def sample_data(
        self, ratings_df: pd.DataFrame, movies_df: pd.DataFrame, users_df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        if not self.sample_users:
            logger.info("No sampling of users")
            return ratings_df, movies_df, users_df
        logger.info(f"Sampling {self.sample_users} users for faster processing...")
        sampled_user_ids = np.random.choice(
            users_df["user_id"].unique(),
            size=min(self.sample_users, len(users_df["user_id"].unique())),
            replace=False,
        )
        users_df = users_df[users_df["user_id"].isin(sampled_user_ids)]
        ratings_df = ratings_df[ratings_df["user_id"].isin(sampled_user_ids)]
        rated_movie_ids = ratings_df["movie_id"].unique()
        movies_df = movies_df[movies_df["movie_id"].isin(rated_movie_ids)]
        logger.info(f"After sampling: {len(users_df)} users, {len(movies_df)} movies, {len(ratings_df)} ratings")
        return ratings_df, movies_df, users_df
