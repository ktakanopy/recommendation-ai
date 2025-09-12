import uuid
from datetime import datetime
from typing import Dict, List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# TODO: move all of them to DTO folder


class FilterPage(BaseModel):
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, gt=0, le=1000, description="Maximum number of items to return")


class Token(BaseModel):
    access_token: str
    token_type: str


class RatingSchema(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    timestamp: Optional[datetime] = None


class UserSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None


class Message(BaseModel):
    message: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: EmailStr
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class UserDB(UserSchema):
    id: int


class UserList(BaseModel):
    users: list[UserPublic]


class MovieSchema(BaseModel):
    id: int
    title: str
    genres: List[str]


class MoviePublic(BaseModel):
    id: int
    title: str
    genres: List[str]
    model_config = ConfigDict(from_attributes=True)


class MovieDB(MovieSchema):
    id: uuid.UUID


class MovieList(BaseModel):
    movies: list[MoviePublic]


class RatingPublic(BaseModel):
    id: uuid.UUID
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)


# TODO: check if this schema should be in dto folder
class RatingRequest(BaseModel):
    """Request schema for rating data"""

    id: uuid.UUID
    user_id: int
    movie_id: int
    timestamp: datetime
    rating: float


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
