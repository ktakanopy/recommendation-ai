from dataclasses import dataclass
from typing import List


@dataclass
class Recommendation:
    """Domain model representing a movie recommendation"""

    movie_id: int
    title: str
    genres: str
    similarity_score: float

    @property
    def similarity_percentage(self) -> float:
        """Get similarity as percentage"""
        return self.similarity_score * 100


@dataclass
class RecommendationResult:
    """Domain model for recommendation results"""

    user_id: str
    recommendations: List[Recommendation]
    total_available_movies: int

    @property
    def recommendation_count(self) -> int:
        """Get number of recommendations"""
        return len(self.recommendations)
