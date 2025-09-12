import uuid
from typing import List
from pydantic import Field
from pydantic import BaseModel


class Movie(BaseModel):
    id: int
    title: str
    genres: List[str] = Field(default_factory=list)
