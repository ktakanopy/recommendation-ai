from typing import Dict, List

from pydantic import BaseModel


class RecommendationRequest(BaseModel):
    """Request schema for existing user recommendations"""

    user_id: str
    user_age: float = 25.0
    gender: str = "M"
    num_recommendations: int = 10


class NewUserRecommendationRequest(BaseModel):
    """Request schema for new user recommendations"""

    user_id: int  # ID of the created user
    num_recommendations: int = 10


class GetOnboardingMoviesRequest(BaseModel):
    """Request schema for onboarding movies"""

    num_movies: int = 10


class RecommendationResponse(BaseModel):
    """Response schema for recommendations"""

    movie_id: int
    title: str
    genres: List[str]
    similarity_score: float


class RecommendationResultResponse(BaseModel):
    """Response schema for recommendation results"""

    user_id: str
    recommendations: List[RecommendationResponse]


class OnboardingMovieResponse(BaseModel):
    """Response schema for onboarding movie"""

    movie_id: int
    title: str
    genres: List[str]


class OnboardingMoviesResultResponse(BaseModel):
    """Response schema for onboarding movies results"""

    recommendations: Dict[str, List[OnboardingMovieResponse]]
