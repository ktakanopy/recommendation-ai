
import uuid
from dataclasses import dataclass, field
from typing import List


@dataclass
class Movie:
    id: uuid.UUID
    title: str
    genres: List[str] = field(default_factory=list)
    embedding: List[float] = field(default_factory=list)
