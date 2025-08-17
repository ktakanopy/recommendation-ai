import uuid
from dataclasses import dataclass
from datetime import datetime


@dataclass
class Rating:
    id: uuid.UUID
    user_id: uuid.UUID
    movie_id: uuid.UUID
    timestamp: datetime
    rating: float
