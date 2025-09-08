import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class Movie:
    id: uuid.UUID
    original_id: int
    title: str
    genres: List[str] = field(default_factory=list)
