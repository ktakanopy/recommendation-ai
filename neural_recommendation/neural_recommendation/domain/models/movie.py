import uuid
from dataclasses import dataclass, field
from typing import List
from pydantic import BaseModel


class Movie(BaseModel):
    id: int
    title: str
    genres: List[str] = field(default_factory=list)
