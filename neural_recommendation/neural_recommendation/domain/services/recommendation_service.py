from typing import Any, Dict, List

import torch
import numpy as np
import torch.nn.functional as F

from neural_recommendation.applications.use_cases.deep_learning.ncf_feature_processor import NCFFeatureProcessor
from neural_recommendation.applications.use_cases.deep_learning.candidate_generator import CandidateGenerator
from neural_recommendation.applications.use_cases.deep_learning.cold_start_recommender import ColdStartRecommender
from neural_recommendation.domain.models.deep_learning.ncf_model import NCFModel
from neural_recommendation.domain.models.deep_learning.recommendation import Recommendation, RecommendationResult
from neural_recommendation.infrastructure.logging.logger import Logger

logger = Logger.get_logger(__name__)


class RecommendationService:
    """Domain service for generating movie recommendations using NCF model"""

    def __init__(self, model: NCFModel, feature_service: NCFFeatureProcessor, movie_mappings: Dict[str, Any]):
        self.model = model
        self.feature_service = feature_service
        self.device = next(model.parameters()).device
        self.model.eval()
        self.title_to_idx = movie_mappings.get("title_to_idx", {})
        self.idx_to_title = movie_mappings.get("idx_to_title", {})
        self.all_movie_titles = movie_mappings.get("all_movie_titles", [])
        # Movie ID to title mapping for quick lookup
        self.movie_id_to_title = {v: k for k, v in self.title_to_idx.items()}
        
        # Initialize cold start components
        self.candidate_generator = CandidateGenerator(
            train_ratings=None,  # Could be loaded from data if available
            movies=None,         # Could be loaded from data if available
            all_movie_ids=list(self.title_to_idx.values())
        )
        
        self.cold_start_recommender = ColdStartRecommender(
            trained_model=model,
            feature_processor=feature_service,
            candidate_generator=self.candidate_generator,
            movies_df=None,  # Could be loaded from data if available
            liked_threshold=4.0
        )

    def generate_recommendations_for_user(
        self,
        user_id: str,
        user_age: float = 25.0,
        gender: str = "M",
        num_recommendations: int = 10,
        batch_size: int = 100,
    ) -> RecommendationResult:
        """Generate recommendations using NCF model"""
        logger.info(f"Generating recommendations for user {user_id}")

        # Process user demographics to get feature vector
        user_demographics = {
            "gender": gender,
            "age": int(user_age),  # Convert to age category
            "occupation": 1,  # Default occupation, could be enhanced
        }

        try:
            user_features = self.feature_service.process_user_demographics(user_demographics)
        except Exception as e:
            logger.error(f"Error processing user demographics: {str(e)}")
            # Fallback to zero features if processing fails
            user_features = torch.zeros(self.feature_service.user_feature_dim or 1)

        # Get all available movies
        available_movie_ids = list(self.title_to_idx.values())
        available_movie_titles = self.all_movie_titles

        # Calculate interaction probabilities for all movies
        probabilities = self._calculate_interaction_probabilities(user_features, available_movie_ids, batch_size)

        # Create recommendations from probabilities
        recommendations = self._create_top_recommendations(
            available_movie_titles, available_movie_ids, probabilities, num_recommendations
        )

        return RecommendationResult(
            user_id=user_id, recommendations=recommendations, total_available_movies=len(available_movie_titles)
        )

    def generate_recommendations_for_new_user(
        self,
        user_age: float = 25.0,
        gender: str = "M",
        occupation: int = 1,
        num_recommendations: int = 10,
    ) -> RecommendationResult:
        """Generate recommendations for a new user using cold start approach"""
        logger.info(f"Generating cold start recommendations for new user: age={user_age}, gender={gender}")
        
        # Prepare user demographics
        user_demographics = {
            'gender': gender,
            'age': int(user_age),
            'occupation': occupation
        }
        
        # Use cold start recommender
        try:
            cold_start_results = self.cold_start_recommender.recommend_for_new_user(
                user_demographics=user_demographics,
                user_ratings=None,  # No previous ratings for new user
                num_recommendations=num_recommendations
            )
            
            # Convert to Recommendation objects
            recommendations = []
            for i, (movie_id, movie_title, score) in enumerate(cold_start_results):
                recommendation = Recommendation(
                    movie_title=movie_title,
                    similarity_score=float(score),
                    genres="Unknown",  # Could be enhanced with actual genres
                    rank=i + 1
                )
                recommendations.append(recommendation)
            
            return RecommendationResult(
                user_id="new_user",
                recommendations=recommendations,
                total_available_movies=len(self.all_movie_titles)
            )
            
        except Exception as e:
            logger.error(f"Error generating cold start recommendations: {str(e)}")
            # Fallback to regular recommendation approach
            return self.generate_recommendations_for_user(
                user_id="new_user",
                user_age=user_age,
                gender=gender,
                num_recommendations=num_recommendations
            )

    def get_onboarding_movies(self, num_movies: int = 10) -> List[Dict[str, Any]]:
        """Get diverse movies for new user onboarding"""
        logger.info(f"Getting {num_movies} onboarding movies")
        
        try:
            onboarding_results = self.cold_start_recommender.get_onboarding_movies(num_movies)
            
            # Convert to dictionary format
            movies = []
            for movie_id, title, genres in onboarding_results:
                movies.append({
                    "movie_id": movie_id,
                    "title": title,
                    "genres": genres
                })
            
            return movies
            
        except Exception as e:
            logger.error(f"Error getting onboarding movies: {str(e)}")
            # Fallback to simple movie list
            movie_ids = list(self.title_to_idx.values())[:num_movies]
            return [
                {
                    "movie_id": movie_id,
                    "title": self.movie_id_to_title.get(movie_id, f"Movie_{movie_id}"),
                    "genres": "Unknown"
                }
                for movie_id in movie_ids
            ]

    def _calculate_interaction_probabilities(
        self, user_features: torch.Tensor, movie_ids: List[int], batch_size: int = 100
    ) -> List[float]:
        """Calculate interaction probabilities between user and movies using NCF model"""
        probabilities = []
        user_features = user_features.to(self.device)

        # Process movies in batches
        for i in range(0, len(movie_ids), batch_size):
            batch_movie_ids = movie_ids[i : i + batch_size]

            # Get movie features for this batch
            batch_movie_features = []
            for movie_id in batch_movie_ids:
                movie_features = self.feature_service.get_movie_features(movie_id)
                batch_movie_features.append(movie_features)

            if not batch_movie_features:
                continue

            # Stack movie features and move to device
            batch_movie_tensor = torch.stack(batch_movie_features).to(self.device)

            # Repeat user features for batch size
            batch_size_actual = len(batch_movie_features)
            batch_user_features = user_features.unsqueeze(0).repeat(batch_size_actual, 1)

            # Get predictions from NCF model
            with torch.no_grad():
                batch_predictions = self.model.predict_batch(batch_user_features, batch_movie_tensor)
                probabilities.extend(batch_predictions.cpu().numpy().tolist())

        return probabilities

    def _create_top_recommendations(
        self, movie_titles: List[str], movie_ids: List[int], probabilities: List[float], num_recommendations: int
    ) -> List[Recommendation]:
        """Create top recommendations from interaction probabilities"""

        # Combine movies with their probabilities
        movie_probability_pairs = list(zip(movie_titles, movie_ids, probabilities))

        # Sort by probability (descending)
        movie_probability_pairs.sort(key=lambda x: x[2], reverse=True)

        # Create recommendation objects
        recommendations = []
        for i, (title, movie_id, probability) in enumerate(movie_probability_pairs[:num_recommendations]):
            recommendation = Recommendation(
                movie_title=title,
                similarity_score=float(probability),
                genres="Unknown",  # Could be enhanced to get actual genres
                rank=i + 1,
            )
            recommendations.append(recommendation)

        return recommendations

    def explain_recommendation(
        self, user_id: str, movie_title: str, user_age: float = 25.0, gender: str = "M"
    ) -> Dict[str, Any]:
        """Explain why a specific movie was recommended using NCF model"""
        logger.info(f"Explaining recommendation for user {user_id} and movie {movie_title}")

        # Process user demographics
        user_demographics = {"gender": gender, "age": int(user_age), "occupation": 1}

        try:
            user_features = self.feature_service.process_user_demographics(user_demographics)
        except Exception as e:
            logger.error(f"Error processing user demographics for explanation: {str(e)}")
            return {
                "movie_title": movie_title,
                "explanation": "Unable to process user demographics",
                "similarity_score": 0.0,
                "similarity_percentage": 0.0,
                "genres": "Unknown",
            }

        # Get movie ID from title
        movie_id = self.title_to_idx.get(movie_title)
        if movie_id is None:
            return {
                "movie_title": movie_title,
                "explanation": "Movie not found in recommendation candidates",
                "similarity_score": 0.0,
                "similarity_percentage": 0.0,
                "genres": "Unknown",
            }

        # Get movie features
        movie_features = self.feature_service.get_movie_features(movie_id)

        # Calculate interaction probability
        user_features = user_features.to(self.device).unsqueeze(0)
        movie_features = movie_features.to(self.device).unsqueeze(0)

        with torch.no_grad():
            probability = self.model.predict_batch(user_features, movie_features)
            similarity_score = float(probability.item())
            similarity_percentage = similarity_score * 100

        return {
            "movie_title": movie_title,
            "similarity_score": similarity_score,
            "similarity_percentage": similarity_percentage,
            "explanation": (
                f"This movie has a {similarity_percentage:.1f}% interaction probability based on your "
                f"demographic profile and the movie's content features."
            ),
            "genres": "Unknown",  # Could be enhanced to get actual genres
        }
