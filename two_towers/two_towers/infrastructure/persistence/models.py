import uuid
from datetime import datetime
from typing import List

from pgvector.sqlalchemy import Vector
from sqlalchemy import func
from sqlalchemy.orm import Mapped, mapped_column, registry, relationship

table_registry = registry()

Base = declarative_base()

N_DIM = 64


@table_registry.mapped_as_dataclass
class Rating:
    __tablename__ = "ratings"

    id: Mapped[uuid.UUID] = mapped_column(init=False, primary_key=True)
    user_id: Mapped[int]
    movie_id: Mapped[uuid.UUID]


@table_registry.mapped_as_dataclass
class User:
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    ratings: Mapped[List[Rating]] = relationship("Rating", back_populates="user")

    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())


@table_registry.mapped_as_dataclass
class Movie:
    __tablename__ = "movies"

    id: Mapped[uuid.UUID] = mapped_column(init=False, primary_key=True)
    title: Mapped[str]
    genres: Mapped[str]
    embedding = Vector(N_DIM)
