from collections import Counter
from typing import Any, Dict, List, Optional, Tuple
import torch
import torch.nn.functional as F

from neural_recommendation.applications.use_cases.deep_learning.candidate_generator import CandidateGenerator
from neural_recommendation.applications.use_cases.deep_learning.ncf_feature_processor import NCFFeatureProcessor
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class ColdStartRecommender:
    """
    Cold Start Recommendation System for new users

    Handles recommendations for users with limited or no interaction history
    by leveraging user demographics, initial ratings, and content-based filtering.
    """

    def __init__(
        self,
        trained_model: NCFModel,
        feature_processor: NCFFeatureProcessor,
        candidate_generator: CandidateGenerator,
        movie_genres_dict: Optional[Dict[str, List[str]]] = None,
        liked_threshold: float = 4.0,
    ):
        self.model = trained_model
        self.feature_processor = feature_processor
        self.candidate_generator = candidate_generator
        self.movie_genres_dict = movie_genres_dict
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.liked_threshold = liked_threshold

    def get_similar_users_neural_embedding(self, user_demographics: Dict[str, Any], top_k: int = 50) -> List[int]:
        """
        Find users with similar demographics using neural network embeddings

        Uses the trained NCF model's user feature processing layers to compute
        user embeddings and find similar users based on cosine similarity.

        Args:
            user_demographics: dict with user demographic info
            top_k: number of similar users to return

        Returns:
            list: user_ids of similar users
        """
        try:
            # Get the new user's feature vector and neural representation
            new_user_features = self.feature_processor.process_user_demographics(user_demographics)
            new_user_features = new_user_features.detach().clone().unsqueeze(0).to(self.device)

            # Get neural representation using the trained model's user processing layers
            self.model.eval()
            with torch.no_grad():
                # Process through the trained neural network layers (same as in NCF forward pass)
                user_x = self.model.dropout(torch.relu(self.model.user_bn1(self.model.user_fc1(new_user_features))))
                new_user_embedding = torch.relu(self.model.user_bn2(self.model.user_fc2(user_x)))

            # Get embeddings for all existing users
            similar_users = []
            existing_user_ids = list(self.feature_processor.user_features_cache.keys())

            if not existing_user_ids:
                logger.warning("No cached user features available for similarity computation")
                return []

            # Process existing users in batches for efficiency
            batch_size = 128
            for i in range(0, len(existing_user_ids), batch_size):
                batch_user_ids = existing_user_ids[i : i + batch_size]
                batch_features = []

                for user_id in batch_user_ids:
                    cached_features = self.feature_processor.user_features_cache[user_id]
                    batch_features.append(cached_features)

                # Convert to tensor and move to device
                batch_features = torch.stack(batch_features).to(self.device)

                # Get neural embeddings for this batch
                with torch.no_grad():
                    batch_x = self.model.dropout(torch.relu(self.model.user_bn1(self.model.user_fc1(batch_features))))
                    batch_embeddings = torch.relu(self.model.user_bn2(self.model.user_fc2(batch_x)))

                    # Compute cosine similarity with the new user
                    similarities = F.cosine_similarity(new_user_embedding, batch_embeddings, dim=1)

                    # Store results
                    for j, user_id in enumerate(batch_user_ids):
                        similarity_score = similarities[j].item()
                        similar_users.append((user_id, similarity_score))

            # Sort by similarity score (descending) and return top_k
            similar_users.sort(key=lambda x: x[1], reverse=True)
            return [user_id for user_id, _ in similar_users[:top_k]]

        except Exception as e:
            logger.error(f"Error computing similar users: {str(e)}")
            return []

    def get_similar_user_candidates(
        self, user_demographics: Dict[str, Any], top_k: int = 10, num_candidates: int = 100
    ) -> List[int]:
        """Get candidate movies from similar users"""
        similar_users = self.get_similar_users_neural_embedding(user_demographics)
        candidates = []

        if similar_users:
            similar_user_movies = []
            for similar_user_id in similar_users[:top_k]:  # Top k similar users
                user_movies = self.candidate_generator.user_interacted_items.get(similar_user_id, [])
                similar_user_movies.extend(user_movies)

            movie_counts = Counter(similar_user_movies)
            demographic_candidates = [movie_id for movie_id, _ in movie_counts.most_common(num_candidates // 2)]
            candidates.extend(demographic_candidates)

        return candidates

    def generate_cold_start_candidates(
        self,
        user_demographics: Dict[str, Any],
        user_ratings: Optional[List[Tuple[int, float]]] = None,
        num_candidates: int = 100,
    ) -> List[int]:
        """
        Generate candidate movies for cold start scenario

        Args:
            user_demographics: dict with user demographic info
            user_ratings: list of (movie_id, rating) tuples for initial ratings
            num_candidates: number of candidates to generate

        Returns:
            list: candidate movie IDs
        """
        candidates = []

        if user_ratings is None or len(user_ratings) == 0:
            # Pure cold start - no ratings yet
            logger.info("Generating pure cold start candidates")

            # Get hybrid candidates (popularity + collaborative + content)
            hybrid_candidates = self.candidate_generator.generate_hybrid_candidates(
                user_id=-1,  # dummy user_id
                num_candidates=num_candidates // 2,
            )
            candidates.extend(hybrid_candidates)

            # Get recommendations based on similar users' preferences (using neural embeddings)
            similar_users = self.get_similar_users_neural_embedding(user_demographics)
            if similar_users:
                similar_user_movies = []
                for similar_user_id in similar_users[:10]:  # Top 10 similar users
                    user_movies = self.candidate_generator.user_interacted_items.get(similar_user_id, [])
                    similar_user_movies.extend(user_movies)

                movie_counts = Counter(similar_user_movies)
                demographic_candidates = [movie_id for movie_id, _ in movie_counts.most_common(num_candidates // 2)]
                candidates.extend(demographic_candidates)

        else:
            # Warm start - user has some ratings
            logger.info(f"Generating warm start candidates with {len(user_ratings)} ratings")

            liked_movies = [movie_id for movie_id, rating in user_ratings if rating >= self.liked_threshold]

            user_similar_candidates = self.get_similar_user_candidates(
                user_demographics, top_k=10, num_candidates=num_candidates // 3
            )
            all_ids = getattr(self.candidate_generator, "all_movieIds", getattr(self.candidate_generator, "all_movie_ids", [])) # TODO: create a function inside candidate generator
            available_items = set(all_ids) - set(liked_movies)
            popular_candidates = self.candidate_generator.generate_popularity_candidates(
                user_id=-1,
                num_candidates=num_candidates // 3,
                user_available_items=available_items,
            )
            user_genres = self.candidate_generator.get_genres_from_movies(liked_movies)
            content_candidates = self.candidate_generator.generate_content_candidates(
                user_id=-1,
                num_candidates=num_candidates // 3,
                user_available_items=available_items,
                passed_user_genres=user_genres,
            )
            candidates.extend(user_similar_candidates)
            candidates.extend(popular_candidates)
            candidates.extend(content_candidates)

        # Remove duplicates while preserving order
        seen = set()
        unique_candidates = []
        for movie_id in candidates:
            if movie_id not in seen:
                seen.add(movie_id)
                unique_candidates.append(movie_id)

        return unique_candidates[:num_candidates]

    def recommend_for_new_user(
        self,
        user_demographics: Dict[str, Any],
        user_ratings: Optional[List[Tuple[int, float]]] = None,
        num_recommendations: int = 10,
    ) -> List[Tuple[int, str, float]]:
        """
        Generate recommendations for a new user

        Args:
            user_demographics: dict with keys 'gender', 'age', 'occupation'
            user_ratings: list of (movie_id, rating) tuples for initial ratings (optional)
            num_recommendations: number of recommendations to return

        Returns:
            list: list of (movie_id, title, predicted_score) tuples
        """
        logger.info(f"Generating recommendations for new user with demographics: {user_demographics}")

        # Create user feature vector using FeatureProcessor
        try:
            user_features = self.feature_processor.process_user_demographics(user_demographics) # TODO: check if those are only features we need
            user_features = user_features.detach().clone().unsqueeze(0).to(self.device)  # Add batch dimension
        except Exception as e:
            logger.error(f"Error processing user demographics: {str(e)}")
            # Return empty recommendations if feature processing fails
            return []

        # Generate candidate movies
        candidates = self.generate_cold_start_candidates(
            user_demographics,
            user_ratings,
            num_candidates=min(
                200,
                len(getattr(self.candidate_generator, "all_movieIds", getattr(self.candidate_generator, "all_movie_ids", [])))
            ),
        )

        if not candidates:
            logger.warning("No candidates generated for cold start recommendation")
            return []

        # Score candidates using the NCF model
        movie_scores = []

        self.model.eval()
        with torch.no_grad():
            for movie_id in candidates:
                try:
                    # Get movie features
                    movie_features = self.feature_processor.get_movie_features(movie_id)
                    movie_features = movie_features.unsqueeze(0).to(self.device)

                    # Predict score using NCF model
                    score = self.model(user_features, movie_features).item()

                    # Get movie title
                    movie_title = f"Movie_{movie_id}"

                    movie_scores.append((movie_id, movie_title, score))

                except Exception as e:
                    logger.warning(f"Error processing movie {movie_id}: {str(e)}")
                    continue

        # Sort by predicted score and return top recommendations
        movie_scores.sort(key=lambda x: x[2], reverse=True)

        # Filter out movies user has already rated
        if user_ratings:
            rated_movie_ids = {movie_id for movie_id, _ in user_ratings}
            movie_scores = [(mid, title, score) for mid, title, score in movie_scores if mid not in rated_movie_ids]

        logger.info(f"Generated {len(movie_scores)} cold start recommendations")
        return movie_scores[:num_recommendations]

    def get_onboarding_movies(self, num_movies: int = 10) -> List[Tuple[int, str, str]]:
        """
        Get diverse, popular movies for new user onboarding/rating collection

        Args:
            num_movies: number of movies to return for rating

        Returns:
            list: list of (movie_id, title, genres) tuples
        """
        logger.info(f"Getting {num_movies} onboarding movies")
        all_ids = getattr(self.candidate_generator, "all_movieIds", getattr(self.candidate_generator, "all_movie_ids", []))
        available_items = set(all_ids)
        popular_movies = self.candidate_generator.generate_popularity_candidates(user_id=-1, user_available_items=available_items, num_candidates=max(200, num_movies * 10))

        if not popular_movies:
            logger.warning("No popular movies available for onboarding")
            return []

        if hasattr(self.candidate_generator, "all_genres") and self.candidate_generator.all_genres:
            genres = self.candidate_generator.all_genres
        else:
            genres = list({g for gs in self.candidate_generator.movie_to_genres.values() for g in gs})

        selected_movies = []
        used_movie_ids = set()

        for genre in genres:
            if len(selected_movies) >= num_movies:
                break
            for movie_id in popular_movies:
                if movie_id in used_movie_ids:
                    continue
                if movie_id in self.candidate_generator.movie_to_genres and genre in self.candidate_generator.movie_to_genres[movie_id]:
                    movie_genres = self.candidate_generator.movie_to_genres[movie_id]
                    movie_title = f"Movie_{movie_id}"
                    movie_genres_str = "|".join(movie_genres)
                    selected_movies.append((movie_id, movie_title, movie_genres_str))
                    used_movie_ids.add(movie_id)
                    break

        if len(selected_movies) < num_movies:
            for movie_id in popular_movies:
                if len(selected_movies) >= num_movies:
                    break
                if movie_id in used_movie_ids:
                    continue
                if movie_id in self.candidate_generator.movie_to_genres:
                    movie_genres = self.candidate_generator.movie_to_genres[movie_id]
                    movie_title = f"Movie_{movie_id}"
                    movie_genres_str = "|".join(movie_genres)
                    selected_movies.append((movie_id, movie_title, movie_genres_str))
                else:
                    movie_title = f"Movie_{movie_id}"
                    selected_movies.append((movie_id, movie_title, "Unknown"))
                used_movie_ids.add(movie_id)

        logger.info(f"Selected {len(selected_movies)} diverse onboarding movies")
        return selected_movies[:num_movies]
