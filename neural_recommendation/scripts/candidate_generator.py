from functools import lru_cache
from collections import defaultdict
import pickle
import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

from ncf_feature_processor import NCFFeatureProcessor


# global data structures for fast candidate generation
class CandidateGenerator:
    def __init__(self, all_movie_ids, train_ratings=None, movies=None, feature_processor: NCFFeatureProcessor = None):
        self.train_ratings = train_ratings
        self.movies = movies
        self.all_movie_ids = all_movie_ids
        self.feature_processor = feature_processor

        self.user_interacted_items = (
            train_ratings.groupby("user_id")["movie_id"].apply(list).to_dict() if train_ratings is not None else None
        )

        # Pre-compute global data structures
        self._precompute_popularity()
        self._precompute_movie_genres()
        self._precompute_top_popular_movies_by_genres()

    def _precompute_popularity(self):
        """Pre-compute item popularity ranking"""
        self.item_popularity = self.train_ratings["movie_id"].value_counts()
        self.popular_items = self.item_popularity.index.tolist()

    def _precompute_movie_genres(self):
        """Create movie-to-genres dictionary for O(1) lookups"""
        self.movie_to_genres = {}
        for _, row in self.movies.iterrows():
            self.movie_to_genres[row["movie_id"]] = row["genres"].split("|")

    def save_popularity(self, dir_path: str, filepath: str):
        with open(os.path.join(dir_path, filepath), "wb") as f:
            pickle.dump(self.popular_items, f, protocol=pickle.HIGHEST_PROTOCOL)

    def save_top_popular_movies_by_genres(self, dir_path: str, filepath: str):
        with open(os.path.join(dir_path, filepath), "wb") as f:
            pickle.dump(self.top_popular_movies_by_genre, f, protocol=pickle.HIGHEST_PROTOCOL)

    def load_popularity(self, dir_path: str, filepath: str):
        with open(os.path.join(dir_path, filepath), "rb") as f:
            self.popular_items = pickle.load(f)

    def get_genres_from_movies(self, movies_ids):
        genre_counts = {}
        for movie_id in movies_ids:
            if movie_id in self.movie_to_genres:
                for genre in self.movie_to_genres[movie_id]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return genre_counts

    def _precompute_top_popular_movies_by_genres(self):
        """Pre-compute top popular movies by genres"""
        genre_to_items = defaultdict(list)
        if hasattr(self, "movie_to_genres") and self.movie_to_genres:
            items_iter = self.movie_to_genres.items()
        elif getattr(self, "movies", None) is not None:
            temp = {}
            for _, row in self.movies.iterrows():
                genres = row["genres"].split("|") if isinstance(row["genres"], str) else []
                temp[row["movie_id"]] = genres
            items_iter = temp.items()
        else:
            self.top_popular_movies_by_genre = {}
            return

        get_popularity = self.item_popularity.get if hasattr(self, "item_popularity") else lambda _: 0

        for movie_id, genres in items_iter:
            popularity = int(get_popularity(movie_id, 0)) if callable(get_popularity) else 0
            for genre in genres:
                genre_to_items[genre].append((movie_id, popularity))

        result = {}
        for genre, items in genre_to_items.items():
            items.sort(key=lambda x: x[1], reverse=True)
            result[genre] = [movie_id for movie_id, _ in items]
        self.top_popular_movies_by_genre = result

    @lru_cache(maxsize=1000)
    def get_available_items(self, user_id):
        """Cache available items for users"""
        interacted_items = set(self.user_interacted_items.get(user_id, []))
        all_items_set = set(self.all_movie_ids)
        return list(all_items_set - interacted_items)

    def generate_popularity_candidates(self, user_id, user_available_items=None, num_candidates=100):
        """Optimized popularity-based candidate generation"""
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

    def generate_collaborative_candidates(self, user_id, num_candidates=100):
        """Optimized collaborative filtering using Annoy-based similar users when available"""
        available_items = set(self.get_available_items(user_id))

        similar_user_ids = self.feature_processor.get_similar_users(user_id, top_k=num_candidates)
        candidate_scores = defaultdict(float)
        for similar_user_id in similar_user_ids:
            for item in self.user_interacted_items.get(similar_user_id, []):
                if item in available_items:
                    candidate_scores[item] += 1.0

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:num_candidates]]

    def generate_content_candidates(
        self,
        user_id,
        num_candidates=100,
    ):
        available_items = set(self.get_available_items(user_id))
        interacted = self.user_interacted_items.get(user_id, []) or []
        if not interacted:
            return list(available_items)[:num_candidates]

        candidate_scores = defaultdict(float)
        for item in interacted:
            try:
                similars = self.feature_processor.get_similar_movies(item, top_k=num_candidates)
            except Exception:
                similars = []
            for m in similars:
                if m in available_items:
                    candidate_scores[m] += 1.0

        if not candidate_scores:
            return list(available_items)[:num_candidates]

        sorted_candidates = sorted(candidate_scores.items(), key=lambda x: x[1], reverse=True)
        return [item for item, score in sorted_candidates[:num_candidates]]

    def generate_hybrid_candidates(self, user_id, num_candidates=100):
        """Optimized hybrid candidate generation"""
        # Get candidates from each method
        pop_candidates = self.generate_popularity_candidates(user_id, num_candidates=num_candidates // 3)
        collab_candidates = self.generate_collaborative_candidates(user_id, num_candidates=num_candidates // 3)
        content_candidates = self.generate_content_candidates(user_id, num_candidates=num_candidates // 3)

        # Combine and deduplicate using set operations
        hybrid_candidates = list(dict.fromkeys(pop_candidates + collab_candidates + content_candidates))

        # Fill remaining slots with random items
        remaining_slots = num_candidates - len(hybrid_candidates)
        if remaining_slots > 0:
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

    def generate_candidates(self, user_id, method="hybrid", num_candidates=100):
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
            indices = np.random.choice(
                len(available_items),
                size=min(num_candidates, len(available_items)),
                replace=False,
            )
            return [available_items[i] for i in indices]

    def generate_validation_candidates(self, user_id, training_ratings, method="hybrid", num_candidates=100):
        """
        Generate candidates for validation, excluding items the user has already rated in training

        Args:
            user_id: User ID
            training_ratings: DataFrame containing training ratings
            method: Candidate generation method
            num_candidates: Number of candidates to generate

        Returns:
            List of candidate movie IDs excluding training items
        """
        # Get user's training interactions
        user_training_items = set(  # TODO: we could use user_interacted_items instead of training_ratings
            training_ratings[training_ratings["user_id"] == user_id]["movie_id"].tolist()
        )

        # Generate candidates using the specified method
        if method == "popularity":
            candidates = self.generate_popularity_candidates(user_id, num_candidates=num_candidates * 2)
        elif method == "collaborative":
            candidates = self.generate_collaborative_candidates(user_id, num_candidates=num_candidates * 2)
        elif method == "content":
            candidates = self.generate_content_candidates(user_id, num_candidates=num_candidates * 2)
        elif method == "hybrid":
            candidates = self.generate_hybrid_candidates(user_id, num_candidates=num_candidates * 2)
        else:
            # Random fallback
            available_items = self.get_available_items(user_id)
            candidates = available_items[: num_candidates * 2]

        # Filter out training items
        validation_candidates = [item for item in candidates if item not in user_training_items]

        # Return requested number of candidates (or all available if fewer)
        return validation_candidates[:num_candidates]

    def precompute_training_candidates(self, training_ratings, method="hybrid", num_candidates=100):
        """
        Precompute training candidates for all users in training set
        """
        users = training_ratings["user_id"].unique()
        training_candidates = {}

        print(f"Precomputing training candidates for {len(users)} users...")

        for user_id in tqdm(users, desc="Precomputing training candidates"):
            candidates = self.generate_candidates(user_id, method, num_candidates)
            training_candidates[user_id] = candidates

        print(f"Precomputed training candidates for {len(training_candidates)} users")
        return training_candidates

    def precompute_validation_candidates(
        self, validation_ratings, training_ratings, method="hybrid", num_candidates=100
    ):
        """
        Precompute validation candidates for all users in validation set, excluding training items

        Args:
            validation_ratings: DataFrame containing validation ratings
            training_ratings: DataFrame containing training ratings
            method: Candidate generation method
            num_candidates: Number of candidates per user

        Returns:
            Dict of {user_id: [candidate_items]} for validation
        """
        users = validation_ratings["user_id"].unique()
        validation_candidates = {}

        print(f"Precomputing validation candidates for {len(users)} users...")

        for user_id in tqdm(users, desc="Precomputing validation candidates"):
            candidates = self.generate_validation_candidates(user_id, training_ratings, method, num_candidates)
            validation_candidates[user_id] = candidates

        print(f"Precomputed validation candidates for {len(validation_candidates)} users")
        return validation_candidates

    def save(self, dir_path: str, filepath: str):
        data = {
            "version": 1,
            "all_movie_ids": self.all_movie_ids,
            "user_interacted_items": self.user_interacted_items,
            "popular_items": getattr(self, "popular_items", None),
            "movie_to_genres": getattr(self, "movie_to_genres", None),
            "user_similarity": getattr(self, "user_similarity", None),
            "user_to_idx": getattr(self, "user_to_idx", None),
            "idx_to_user": getattr(self, "idx_to_user", None),
            "user_genre_profiles": getattr(self, "user_genre_profiles", None),
            "all_genres": getattr(self, "all_genres", None),
            "genre_to_idx": getattr(self, "genre_to_idx", None),
        }
        with open(os.path.join(dir_path, filepath), "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, dir_path: str, filepath: str):
        with open(os.path.join(dir_path, filepath), "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.train_ratings = None
        obj.movies = None
        obj.all_movie_ids = data.get("all_movie_ids")
        obj.user_interacted_items = data.get("user_interacted_items")
        obj.popular_items = data.get("popular_items")
        obj.movie_to_genres = data.get("movie_to_genres")
        return obj
