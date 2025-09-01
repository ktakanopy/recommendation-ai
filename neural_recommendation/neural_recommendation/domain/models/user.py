from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from neural_recommendation.domain.models.rating import Rating


@dataclass
class User:
    username: str
    email: str
    password_hash: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    ratings: Optional[List[Rating]] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
