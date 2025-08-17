from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional

from two_towers.domain.models.rating import Rating


@dataclass
class User:
    username: str
    age: Optional[int] = None
    gender: Optional[str] = None
    occupation: Optional[str] = None
    email: str
    password_hash: str
    id: Optional[int] = None
    created_at: Optional[datetime] = None
    ratings: List[Rating] = field(default_factory=list)

    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.now()
