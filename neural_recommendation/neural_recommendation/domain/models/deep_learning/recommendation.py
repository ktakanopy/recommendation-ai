from typing import List, Optional

from pydantic import BaseModel


class Recommendation(BaseModel):
    """Domain model representing a movie recommendation"""

    movie_id: int
    title: str
    genres: List[str]
    similarity_score: Optional[float] = None

    @property
    def similarity_percentage(self) -> float:
        """Get similarity as percentage"""
        return self.similarity_score * 100


class RecommendationResult(BaseModel):
    """Domain model for recommendation results"""

    user_id: str
    recommendations: List[Recommendation]
