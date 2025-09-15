from functools import lru_cache
from collections import defaultdict
import pickle
import os
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm


# global data structures for fast candidate generation
class CandidateGenerator:
    def __init__(self, train_ratings, movies, all_movieIds):
        self.train_ratings = train_ratings
        self.movies = movies
        self.all_movieIds = all_movieIds
        self.user_interacted_items = (
            train_ratings.groupby("user_id")["movie_id"].apply(list).to_dict()
        )

        # Pre-compute global data structures
        self._precompute_popularity()
        self._precompute_movie_genres()
        self._precompute_user_similarity()
        self._precompute_genre_profiles()

    def _precompute_popularity(self):
        """Pre-compute item popularity ranking"""
        self.item_popularity = self.train_ratings["movie_id"].value_counts()
        self.popular_items = self.item_popularity.index.tolist()


    def _precompute_movie_genres(self):
        """Create movie-to-genres dictionary for O(1) lookups"""
        self.movie_to_genres = {}
        for _, row in self.movies.iterrows():
            self.movie_to_genres[row["movie_id"]] = row["genres"].split("|")

    def _precompute_user_similarity(self):
        """Pre-compute user similarity matrix using vectorized operations"""
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
        self.user_item_matrix = csr_matrix(
            (data, (rows, cols)), shape=(len(users), max(self.all_movieIds) + 1)
        )

        # Compute user similarity (cosine similarity for efficiency)
        self.user_similarity = cosine_similarity(self.user_item_matrix)
        self.user_to_idx = user_to_idx
        self.idx_to_user = {idx: user for user, idx in user_to_idx.items()}

    def _precompute_genre_profiles(self):
        """Pre-compute genre profiles for users"""
        self.user_genre_profiles = {}
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

    def get_genres_from_movies(self, movies_ids):
        genre_counts = {}
        for movie_id in movies_ids:
            if movie_id in self.movie_to_genres:
                for genre in self.movie_to_genres[movie_id]:
                    genre_counts[genre] = genre_counts.get(genre, 0) + 1
        return genre_counts

    @lru_cache(maxsize=1000)
    def get_available_items(self, user_id):
        """Cache available items for users"""
        interacted_items = set(self.user_interacted_items.get(user_id, []))
        all_items_set = set(self.all_movieIds)
        return list(all_items_set - interacted_items)

    def generate_popularity_candidates(
        self, user_id, user_available_items=None, num_candidates=100
    ):
        """Optimized popularity-based candidate generation"""
        available_items = (
            set(self.get_available_items(user_id))
            if user_available_items is None
            else user_available_items
        )

        candidates = []
        for item in self.popular_items:
            if item in available_items:
                candidates.append(item)
                if len(candidates) >= num_candidates:
                    break

        return candidates[:num_candidates]

    def generate_collaborative_candidates(self, user_id, num_candidates=100):
        """Optimized collaborative filtering using pre-computed similarity"""
        if user_id not in self.user_to_idx:
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
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [item for item, score in sorted_candidates[:num_candidates]]

    def generate_content_candidates(
        self,
        user_id,
        passed_user_genres=None,
        user_available_items=None,
        num_candidates=100,
    ):
        """Optimized content-based filtering using pre-computed genre profiles"""
        user_genres = (
            self.user_genre_profiles.get(user_id, {})
            if passed_user_genres is None
            else passed_user_genres
        )
        if not user_genres:
            return self.get_available_items(user_id)[:num_candidates]

        available_items = (
            set(self.get_available_items(user_id))
            if user_available_items is None
            else user_available_items
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
        sorted_candidates = sorted(
            candidate_scores.items(), key=lambda x: x[1], reverse=True
        )
        return [item for item, score in sorted_candidates[:num_candidates]]

    def generate_hybrid_candidates(self, user_id, num_candidates=100):
        """Optimized hybrid candidate generation"""
        # Get candidates from each method
        pop_candidates = self.generate_popularity_candidates(
            user_id, num_candidates=num_candidates // 3
        )
        collab_candidates = self.generate_collaborative_candidates(
            user_id, num_candidates=num_candidates // 3
        )
        content_candidates = self.generate_content_candidates(
            user_id, num_candidates=num_candidates // 3
        )

        # Combine and deduplicate using set operations
        hybrid_candidates = list(
            dict.fromkeys(pop_candidates + collab_candidates + content_candidates)
        )

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
            return self.generate_popularity_candidates(
                user_id, num_candidates=num_candidates
            )
        elif method == "collaborative":
            return self.generate_collaborative_candidates(
                user_id, num_candidates=num_candidates
            )
        elif method == "content":
            return self.generate_content_candidates(
                user_id, num_candidates=num_candidates
            )
        elif method == "hybrid":
            return self.generate_hybrid_candidates(
                user_id, num_candidates=num_candidates
            )
        else:
            # Random fallback
            available_items = self.get_available_items(user_id)
            indices = np.random.choice(
                len(available_items),
                size=min(num_candidates, len(available_items)),
                replace=False,
            )
            return [available_items[i] for i in indices]

    def generate_validation_candidates(
        self, user_id, training_ratings, method="hybrid", num_candidates=100
    ):
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
        user_training_items = set(
            training_ratings[training_ratings["user_id"] == user_id][
                "movie_id"
            ].tolist()
        )

        # Generate candidates using the specified method
        if method == "popularity":
            candidates = self.generate_popularity_candidates(
                user_id, num_candidates=num_candidates * 2
            )
        elif method == "collaborative":
            candidates = self.generate_collaborative_candidates(
                user_id, num_candidates=num_candidates * 2
            )
        elif method == "content":
            candidates = self.generate_content_candidates(
                user_id, num_candidates=num_candidates * 2
            )
        elif method == "hybrid":
            candidates = self.generate_hybrid_candidates(
                user_id, num_candidates=num_candidates * 2
            )
        else:
            # Random fallback
            available_items = self.get_available_items(user_id)
            candidates = available_items[: num_candidates * 2]

        # Filter out training items
        validation_candidates = [
            item for item in candidates if item not in user_training_items
        ]

        # Return requested number of candidates (or all available if fewer)
        return validation_candidates[:num_candidates]

    def precompute_training_candidates(
        self, training_ratings, method="hybrid", num_candidates=100
    ):
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
            candidates = self.generate_validation_candidates(
                user_id, training_ratings, method, num_candidates
            )
            validation_candidates[user_id] = candidates

        print(
            f"Precomputed validation candidates for {len(validation_candidates)} users"
        )
        return validation_candidates

    def save(self, filepath):
        data = {
            "version": 1,
            "all_movieIds": self.all_movieIds,
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
        directory = os.path.dirname(filepath)
        if directory and not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        with open(filepath, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)

    @classmethod
    def load(cls, filepath):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.train_ratings = None
        obj.movies = None
        obj.all_movieIds = data.get("all_movieIds")
        obj.user_interacted_items = data.get("user_interacted_items")
        obj.popular_items = data.get("popular_items")
        obj.movie_to_genres = data.get("movie_to_genres")
        obj.user_similarity = data.get("user_similarity")
        obj.user_to_idx = data.get("user_to_idx")
        obj.idx_to_user = data.get("idx_to_user")
        obj.user_genre_profiles = data.get("user_genre_profiles")
        obj.all_genres = data.get("all_genres")
        obj.genre_to_idx = data.get("genre_to_idx")
        return obj
