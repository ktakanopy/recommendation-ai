from typing import Optional
from pydantic import BaseModel, ConfigDict
from datetime import datetime
import uuid

class RatingSchema(BaseModel):
    user_id: int
    movie_id: int
    rating: float
    timestamp: Optional[datetime] = None


class RatingPublic(BaseModel):
    id: uuid.UUID
    user_id: int
    movie_id: int
    rating: float
    timestamp: datetime
    model_config = ConfigDict(from_attributes=True)


class RatingRequest(BaseModel):
    """Request schema for rating data"""

    id: uuid.UUID
    user_id: int
    movie_id: int
    timestamp: datetime
    rating: float
