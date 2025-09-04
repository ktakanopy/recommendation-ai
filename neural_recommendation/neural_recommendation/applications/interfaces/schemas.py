import uuid
from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, ConfigDict, EmailStr, Field

# presentation layer dtos


class FilterPage(BaseModel):
    offset: int = Field(default=0, ge=0, description="Number of items to skip")
    limit: int = Field(default=100, gt=0, le=1000, description="Maximum number of items to return")


class Token(BaseModel):
    access_token: str
    token_type: str


class RatingSchema(BaseModel):
    """Schema for rating data"""

    movie_id: int  # Will be converted to UUID
    rating: float
    timestamp: Optional[datetime] = None


class UserSchema(BaseModel):
    username: str
    email: EmailStr
    password: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None
    ratings: Optional[List[RatingSchema]] = None


class Message(BaseModel):
    message: str


class UserPublic(BaseModel):
    id: int
    username: str
    email: EmailStr
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None
    ratings_count: Optional[int] = None
    model_config = ConfigDict(from_attributes=True)


class UserDB(UserSchema):
    id: int


class UserList(BaseModel):
    users: list[UserPublic]


class MovieSchema(BaseModel):
    title: str
    genres: List[str]
    embedding: Optional[List[float]] = None


class MoviePublic(BaseModel):
    id: uuid.UUID
    title: str
    genres: List[str]
    model_config = ConfigDict(from_attributes=True)


class MovieDB(MovieSchema):
    id: uuid.UUID


class MovieList(BaseModel):
    movies: list[MoviePublic]
