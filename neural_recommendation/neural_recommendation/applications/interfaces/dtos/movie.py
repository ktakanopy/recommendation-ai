from typing import List
from pydantic import BaseModel, ConfigDict


class MovieSchema(BaseModel):
    id: int
    title: str
    genres: List[str]


class MoviePublic(BaseModel):
    id: int
    title: str
    genres: List[str]
    model_config = ConfigDict(from_attributes=True)


class MovieList(BaseModel):
    movies: list[MoviePublic]
