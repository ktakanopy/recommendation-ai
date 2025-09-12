import uuid
from pydantic import BaseModel
from datetime import datetime


class Rating(BaseModel):
    id: uuid.UUID
    user_id: int
    movie_id: int
    timestamp: datetime
    rating: float
