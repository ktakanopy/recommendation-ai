from datetime import datetime
from typing import Any, Dict

import numpy as np


class FeatureEngineeringService:
    """Domain service for feature engineering logic - pure business rules"""

    @staticmethod
    def create_timestamp_features(dt: datetime) -> Dict[str, Any]:
        """Create timestamp features for a single datetime - pure function"""
        hour = dt.hour
        day_of_week = dt.weekday()
        month = dt.month
        year = dt.year

        return {
            "hour": hour,
            "day_of_week": day_of_week,
            "day_of_month": dt.day,
            "month": month,
            "year": year,
            "quarter": (month - 1) // 3 + 1,
            # Cyclical encodings
            "hour_sin": np.sin(2 * np.pi * hour / 24),
            "hour_cos": np.cos(2 * np.pi * hour / 24),
            "day_of_week_sin": np.sin(2 * np.pi * day_of_week / 7),
            "day_of_week_cos": np.cos(2 * np.pi * day_of_week / 7),
            "month_sin": np.sin(2 * np.pi * month / 12),
            "month_cos": np.cos(2 * np.pi * month / 12),
            # Binary indicators
            "is_weekend": int(day_of_week >= 5),
            "is_weekday": int(day_of_week < 5),
            "is_morning": int(6 <= hour < 12),
            "is_afternoon": int(12 <= hour < 18),
            "is_evening": int(18 <= hour < 22),
            "is_night": int(hour >= 22 or hour < 6),
            "is_spring": int(3 <= month <= 5),
            "is_summer": int(6 <= month <= 8),
            "is_fall": int(9 <= month <= 11),
            "is_winter": int(month == 12 or month <= 2),
            "is_business_hours": int(9 <= hour <= 17 and day_of_week < 5),
        }

    @staticmethod
    def normalize_age(age: float, age_mean: float, age_std: float) -> float:
        """Normalize age using pre-computed statistics"""
        return (age - age_mean) / age_std

    @staticmethod
    def encode_gender(gender: str) -> int:
        """Encode gender as integer"""
        return 1 if gender == "M" else 0

    @staticmethod
    def get_timestamp_feature_columns() -> list[str]:
        """Get list of all timestamp feature column names"""
        return [
            "hour",
            "day_of_week",
            "day_of_month",
            "month",
            "year",
            "quarter",
            "hour_sin",
            "hour_cos",
            "day_of_week_sin",
            "day_of_week_cos",
            "month_sin",
            "month_cos",
            "is_weekend",
            "is_weekday",
            "is_morning",
            "is_afternoon",
            "is_evening",
            "is_night",
            "is_spring",
            "is_summer",
            "is_fall",
            "is_winter",
            "is_business_hours",
        ]

    @staticmethod
    def compute_statistics(data_series, feature_name: str) -> Dict[str, float]:
        """Compute mean and std for a feature"""
        return {f"{feature_name}_mean": data_series.mean(), f"{feature_name}_std": data_series.std()}
