import uuid
from datetime import datetime

from pydantic import BaseModel


class Rating(BaseModel):
    id: uuid.UUID
    user_id: int
    movie_id: int
    timestamp: datetime
    rating: float
