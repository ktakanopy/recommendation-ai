import time
from typing import Any, Dict, List, Optional

import pandas as pd

from neural_recommendation.applications.interfaces.dtos.feature_info_dto import FeatureInfoDto
from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.services.feature_engineering_service import FeatureEngineeringService


class FeaturePreparationService:
    """
    Application service for preparing features for users in production.

    Main method: prepare_user_features(user_id, ratings, user_age, gender)
    - Takes a List[Rating] to build user features from rating history
    - Uses the most recent rating's timestamp for temporal features
    - Provides user profile analysis through extract_user_profile_from_ratings()

    Legacy method: prepare_user_features_legacy() for backward compatibility
    """

    def __init__(self, feature_info: FeatureInfoDto):
        self.feature_info = feature_info
        self.feature_engineering_service = FeatureEngineeringService()

        # Pre-loaded mappings and statistics (computed at training time)
        self.age_mean = feature_info.age_mean
        self.age_std = feature_info.age_std
        self.timestamp_stats = feature_info.timestamp_stats
        self.user_id_to_idx = {uid: idx for idx, uid in enumerate(feature_info.unique_user_ids)}

    def prepare_user_features(
        self, user_id: str, ratings: List[Rating], user_age: float = 25.0, gender: str = "M", occupation: int = 0
    ) -> Dict[str, Any]:
        """Prepare features for a user based on their rating history"""

        # Use domain service for feature engineering
        normalized_age = self.feature_engineering_service.normalize_age(user_age, self.age_mean, self.age_std)
        gender_encoded = self.feature_engineering_service.encode_gender(gender)

        # Get user index
        user_idx = self.user_id_to_idx.get(user_id, 0)

        return {
            "user_id": user_idx,
            "user_age": normalized_age,
            "gender": gender_encoded,
            "occupation": occupation,
        }

    def extract_user_profile_from_ratings(self, ratings: List[Rating]) -> Dict[str, Any]:
        """Extract user profile information from rating history"""
        if not ratings:
            return {
                "total_ratings": 0,
                "average_rating": 0.0,
                "rating_variance": 0.0,
                "latest_timestamp": time.time(),
                "rating_frequency": 0.0,
            }

        # Basic statistics
        rating_values = [r.rating for r in ratings]
        timestamps = [r.timestamp.timestamp() for r in ratings]

        total_ratings = len(ratings)
        average_rating = sum(rating_values) / total_ratings
        rating_variance = sum((r - average_rating) ** 2 for r in rating_values) / total_ratings
        latest_timestamp = max(timestamps)

        # Calculate rating frequency (ratings per day)
        if total_ratings > 1:
            time_span_days = (max(timestamps) - min(timestamps)) / (24 * 3600)
            rating_frequency = total_ratings / max(time_span_days, 1.0)
        else:
            rating_frequency = 1.0

        return {
            "total_ratings": total_ratings,
            "average_rating": average_rating,
            "rating_variance": rating_variance,
            "latest_timestamp": latest_timestamp,
            "rating_frequency": rating_frequency,
        }

    def prepare_comprehensive_user_features(
        self, user_id: str, ratings: List[Rating], user_age: float = 25.0, gender: str = "M"
    ) -> Dict[str, Any]:
        """
        Prepare comprehensive user features combining basic features and rating history analysis.
        This is the recommended method for production recommendation systems.
        """
        # Get basic user features
        basic_features = self.prepare_user_features(user_id, ratings, user_age, gender)

        # Get user profile from rating history
        profile_features = self.extract_user_profile_from_ratings(ratings)

        # Combine all features
        return {**basic_features, **profile_features}

    def prepare_user_features_legacy(
        self, user_id: str, user_age: float = 25.0, gender: str = "M", timestamp: Optional[float] = None
    ) -> Dict[str, Any]:
        """Legacy method - prepare features for a single user with timestamp"""

        # Use current timestamp if not provided
        if timestamp is None:
            timestamp = time.time()

        # Convert timestamp to datetime
        pd.to_datetime(timestamp, unit="s")

        # Use domain service for feature engineering
        normalized_age = self.feature_engineering_service.normalize_age(user_age, self.age_mean, self.age_std)
        gender_encoded = self.feature_engineering_service.encode_gender(gender)

        # Get user index
        user_idx = self.user_id_to_idx.get(user_id, 0)

        return {
            "user_id": user_idx,
            "timestamp": timestamp,
            "user_age": normalized_age,
            "gender": gender_encoded,
        }

    def prepare_movie_features(self, movie_title: str) -> Dict[str, Any]:
        """Prepare features for a movie"""
        title_to_idx = self.feature_info.sentence_embeddings.title_to_idx
        movie_idx = title_to_idx.get(movie_title, 0)

        return {"movie_idx": movie_idx}
