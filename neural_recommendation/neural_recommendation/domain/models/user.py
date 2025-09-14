from datetime import datetime
from typing import List, Optional

from pydantic import BaseModel, Field

from neural_recommendation.domain.models.rating import Rating


class User(BaseModel):
    username: str
    email: str
    password_hash: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[int] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    ratings: Optional[List[Rating]] = Field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
