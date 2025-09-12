from typing import List, Optional
import uuid
from pydantic import BaseModel
from datetime import datetime


class Recommendation(BaseModel):
    """Domain model representing a movie recommendation"""

    movie_id: int
    title: str
    genres: List[str]
    similarity_score: float

    @property
    def similarity_percentage(self) -> float:
        """Get similarity as percentage"""
        return self.similarity_score * 100


class RecommendationResult(BaseModel):
    """Domain model for recommendation results"""

    user_id: str
    recommendations: List[Recommendation]
