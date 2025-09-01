from typing import Any, Dict

import pandas as pd
import torch
from sentence_transformers import SentenceTransformer

from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class SentenceEmbeddingProcessor:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2", device: str = "cpu"):
        self.model_name = model_name
        self.device = device
        self.model = None

    def precompute_embeddings(self, movies_df: pd.DataFrame) -> Dict[str, Any]:
        logger.info(f"Loading sentence transformer model: {self.model_name}")
        self.model = SentenceTransformer(self.model_name, device=self.device)
        all_genres = set()
        for genres_str in movies_df["genres"]:
            all_genres.update(genres_str.split("|"))
        all_genres.update(["Unknown", "Other"])
        unique_genres = sorted(list(all_genres))
        logger.info(f"Computing embeddings for {len(unique_genres)} genres...")
        genre_embeddings = self.model.encode(unique_genres, convert_to_tensor=True, device=self.device, batch_size=64)
        genre_to_embedding = {genre: genre_embeddings[i] for i, genre in enumerate(unique_genres)}
        logger.info(f"Computing movie title embeddings for {len(movies_df)} movies...")
        titles = movies_df["title"].tolist()
        title_embeddings = self.model.encode(titles, convert_to_tensor=True, device=self.device, batch_size=256)
        logger.info("Computing averaged genre embeddings and concatenating...")
        movie_embeddings = {}
        title_to_idx = {title: idx for idx, title in enumerate(titles)}
        for idx, row in movies_df.iterrows():
            title = row["title"]
            genres = row["genres"].split("|")
            title_idx = title_to_idx[title]
            title_embedding = title_embeddings[title_idx]
            genre_embs = []
            for genre in genres:
                if genre in genre_to_embedding:
                    genre_embs.append(genre_to_embedding[genre])
                else:
                    genre_embs.append(genre_to_embedding["Unknown"])
            if genre_embs:
                avg_genre_embedding = torch.stack(genre_embs).mean(dim=0)
            else:
                avg_genre_embedding = genre_to_embedding["Unknown"]
            concatenated_embedding = torch.cat([title_embedding, avg_genre_embedding], dim=0)
            movie_embeddings[title] = concatenated_embedding
        embedding_dim = concatenated_embedding.shape[0]
        logger.info(f"Concatenated embeddings computed! Dimension: {embedding_dim}")
        logger.info(f"   (Title: {title_embedding.shape[0]} + Genre: {avg_genre_embedding.shape[0]} = {embedding_dim})")
        unique_titles = movies_df["title"].unique().tolist()
        title_to_idx = {title: idx for idx, title in enumerate(unique_titles)}
        embedding_matrix = torch.zeros(len(unique_titles), embedding_dim, device=self.device)
        for idx, title in enumerate(unique_titles):
            if title in movie_embeddings:
                embedding_matrix[idx] = movie_embeddings[title]
        return {
            "title_to_idx": title_to_idx,
            "embedding_matrix": embedding_matrix,
            "embedding_dim": embedding_dim,
        }
