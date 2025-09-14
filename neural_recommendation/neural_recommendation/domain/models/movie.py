from typing import List

from pydantic import BaseModel, Field


class Movie(BaseModel):
    id: int
    title: str
    genres: List[str] = Field(default_factory=list)
