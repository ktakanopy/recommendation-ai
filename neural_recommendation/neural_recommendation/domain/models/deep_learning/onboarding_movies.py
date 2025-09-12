from typing import Dict, List
from pydantic import BaseModel

class OnboardingMovie(BaseModel):
    """Domain model representing a movie recommendation"""

    movie_id: int
    title: str
    genres: List[str]


class OnboardingMoviesResult(BaseModel):
    """Domain model for recommendation results"""

    recommendations: Dict[str, List[OnboardingMovie]]