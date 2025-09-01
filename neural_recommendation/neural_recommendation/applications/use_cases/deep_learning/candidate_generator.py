from collections import defaultdict
from functools import lru_cache
from typing import Dict, List, Optional, Set

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity

from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class CandidateGenerator:
    """Optimized candidate generation for NCF recommendations"""

    def __init__(
        self,
        train_ratings: Optional[pd.DataFrame] = None,
        movies: Optional[pd.DataFrame] = None,
        all_movie_ids: Optional[List[int]] = None,
    ):
        self.train_ratings = train_ratings
        self.movies = movies
        self.all_movie_ids = all_movie_ids or []

        # Initialize empty data structures
        self.user_interacted_items: Dict[int, List[int]] = {}
        self.item_popularity = None
        self.popular_items: List[int] = []
        self.movie_to_genres: Dict[int, List[str]] = {}
        self.user_similarity = None
        self.user_to_idx: Dict[int, int] = {}
        self.idx_to_user: Dict[int, int] = {}
        self.user_genre_profiles: Dict[int, Dict[str, int]] = {}
        self.all_genres: List[str] = []
        self.genre_to_idx: Dict[str, int] = {}

        # Pre-compute if data is available
        if train_ratings is not None:
            self._precompute_data()

    def _precompute_data(self):
        """Pre-compute all necessary data structures"""
        logger.info("Pre-computing candidate generation data structures...")

        if self.train_ratings is not None:
            self.user_interacted_items = self.train_ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
            self._precompute_popularity()

        if self.movies is not None:
            self._precompute_movie_genres()

        if self.train_ratings is not None and len(self.user_interacted_items) > 0:
            self._precompute_user_similarity()
            self._precompute_genre_profiles()

        logger.info("Candidate generation pre-computation completed")

    def _precompute_popularity(self):
        """Pre-compute item popularity ranking"""
        if self.train_ratings is None:
            return

        self.item_popularity = self.train_ratings["movie_id"].value_counts()
        self.popular_items = self.item_popularity.index.tolist()
        logger.info(f"Pre-computed popularity for {len(self.popular_items)} items")

    def _precompute_movie_genres(self):
        """Create movie-to-genres dictionary for O(1) lookups"""
        if self.movies is None:
            return

        self.movie_to_genres = {}
        for _, row in self.movies.iterrows():
            if pd.notna(row.get("genres")):
                self.movie_to_genres[row["movie_id"]] = row["genres"].split("|")

        logger.info(f"Pre-computed genres for {len(self.movie_to_genres)} movies")

    def _precompute_user_similarity(self):
        """Pre-compute user similarity matrix using vectorized operations"""
        if not self.user_interacted_items:
            return

        try:
            # Create user-item matrix
            users = list(self.user_interacted_items.keys())
            user_to_idx = {user: idx for idx, user in enumerate(users)}

            # Build sparse matrix
            rows, cols = [], []
            for user_id, items in self.user_interacted_items.items():
                user_idx = user_to_idx[user_id]
                for item in items:
                    rows.append(user_idx)
                    cols.append(item)

            # Create binary user-item matrix
            data = np.ones(len(rows))
            max_item_id = max(self.all_movie_ids) if self.all_movie_ids else max(cols) if cols else 1
            user_item_matrix = csr_matrix((data, (rows, cols)), shape=(len(users), max_item_id + 1))

            # Compute user similarity (cosine similarity for efficiency)
            self.user_similarity = cosine_similarity(user_item_matrix)
            self.user_to_idx = user_to_idx
            self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}

            logger.info(f"Pre-computed user similarity matrix for {len(users)} users")

        except Exception as e:
            logger.warning(f"Failed to compute user similarity: {str(e)}")

    def _precompute_genre_profiles(self):
        """Pre-compute genre profiles for users"""
        if not self.user_interacted_items or not self.movie_to_genres:
            return

        all_genres = set()
        for genres_list in self.movie_to_genres.values():
            all_genres.update(genres_list)

        self.all_genres = list(all_genres)
        self.genre_to_idx = {genre: idx for idx, genre in enumerate(self.all_genres)}

        for user_id, items in self.user_interacted_items.items():
            genre_counts = defaultdict(int)
            for item in items:
                if item in self.movie_to_genres:
                    for genre in self.movie_to_genres[item]:
                        genre_counts[genre] += 1
            self.user_genre_profiles[user_id] = dict(genre_counts)

        logger.info(f"Pre-computed genre profiles for {len(self.user_genre_profiles)} users")

    def get_genres_from_movies(self, movies_ids: List[int]) -> Dict[str, int]:
        """Get genre counts from a list of movie IDs"""
        genre_counts = {}
        for movie_id in movies_ids:
            if movie_id in self.movie_to_genres:
                for genre in self.movie_to_genres[movie_id]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return genre_counts

    @lru_cache(maxsize=1000)
    def get_available_items(self, user_id: int) -> List[int]:
        """Cache available items for users"""
        interacted_items = set(self.user_interacted_items.get(user_id, []))
        all_items_set = set(self.all_movie_ids)
        return list(all_items_set - interacted_items)

    def generate_popularity_candidates(
        self, user_id: int, user_available_items: Optional[Set[int]] = None, num_candidates: int = 100
    ) -> List[int]:
        """Optimized popularity-based candidate generation"""
        if not self.popular_items:
            return self.all_movie_ids[:num_candidates] if self.all_movie_ids else []

        available_items = (
            set(self.get_available_items(user_id)) if user_available_items is None else user_available_items
        )

        candidates = []
        for item in self.popular_items:
            if item in available_items:
                candidates.append(item)
                if len(candidates) >= num_candidates:
                    break

        return candidates[:num_candidates]

    def generate_collaborative_candidates(self, user_id: int, num_candidates: int = 100) -> List[int]:
        """Optimized collaborative filtering using pre-computed similarity"""
        if user_id not in self.user_to_idx or self.user_similarity is None or not self.user_interacted_items:
            return self.get_available_items(user_id)[:num_candidates]

        user_idx = self.user_to_idx[user_id]
        available_items = set(self.get_available_items(user_id))

        # Get most similar users
        similarities = self.user_similarity[user_idx]
        similar_user_indices = np.argsort(similarities)[-51:-1]  # Top 50 similar users

        # Score candidates based on similar users
        candidate_scores = defaultdict(float)
        for similar_idx in similar_user_indices:
            similar_user_id = self.idx_to_user[similar_idx]
            similarity_score = similarities[similar_idx]

            if similarity_score > 0.1:  # Threshold
                for item in self.user_interacted_items.get(similar_user_id, []):
                    if item in available_items:
                        candidate_scores[item] += similarity_score

        # Sort by score and return top candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:num_candidates]]

    def generate_content_candidates(
        self,
        user_id: int,
        passed_user_genres: Optional[Dict[str, int]] = None,
        user_available_items: Optional[Set[int]] = None,
        num_candidates: int = 100,
    ) -> List[int]:
        """Optimized content-based filtering using pre-computed genre profiles"""
        user_genres = self.user_genre_profiles.get(user_id, {}) if passed_user_genres is None else passed_user_genres
        if not user_genres:
            return self.get_available_items(user_id)[:num_candidates]

        available_items = (
            set(self.get_available_items(user_id)) if user_available_items is None else user_available_items
        )

        # Score items by genre overlap
        candidate_scores = {}
        for item in available_items:
            if item in self.movie_to_genres:
                score = 0
                for genre in self.movie_to_genres[item]:
                    score += user_genres.get(genre, 0)
                if score > 0:
                    candidate_scores[item] = score

        # Sort by score and return top candidates
        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:num_candidates]]

    def generate_hybrid_candidates(self, user_id: int, num_candidates: int = 100) -> List[int]:
        """Optimized hybrid candidate generation"""
        # Get candidates from each method
        pop_candidates = self.generate_popularity_candidates(user_id, num_candidates=num_candidates // 3)
        collab_candidates = self.generate_collaborative_candidates(user_id, num_candidates=num_candidates // 3)
        content_candidates = self.generate_content_candidates(user_id, num_candidates=num_candidates // 3)

        # Combine and deduplicate using set operations
        hybrid_candidates = list(dict.fromkeys(pop_candidates + collab_candidates + content_candidates))

        # Fill remaining slots with random items
        remaining_slots = num_candidates - len(hybrid_candidates)
        if remaining_slots > 0 and self.all_movie_ids:
            available_items = set(self.get_available_items(user_id))
            remaining_items = list(available_items - set(hybrid_candidates))
            if remaining_items:
                random_indices = np.random.choice(
                    len(remaining_items),
                    size=min(remaining_slots, len(remaining_items)),
                    replace=False,
                )
                hybrid_candidates.extend([remaining_items[i] for i in random_indices])

        return hybrid_candidates[:num_candidates]

    def generate_candidates(self, user_id: int, method: str = "hybrid", num_candidates: int = 100) -> List[int]:
        """Main interface for candidate generation"""
        if method == "popularity":
            return self.generate_popularity_candidates(user_id, num_candidates=num_candidates)
        elif method == "collaborative":
            return self.generate_collaborative_candidates(user_id, num_candidates=num_candidates)
        elif method == "content":
            return self.generate_content_candidates(user_id, num_candidates=num_candidates)
        elif method == "hybrid":
            return self.generate_hybrid_candidates(user_id, num_candidates=num_candidates)
        else:
            # Random fallback
            available_items = self.get_available_items(user_id)
            if available_items:
                indices = np.random.choice(
                    len(available_items),
                    size=min(num_candidates, len(available_items)),
                    replace=False,
                )
                return [available_items[i] for i in indices]
            return []
