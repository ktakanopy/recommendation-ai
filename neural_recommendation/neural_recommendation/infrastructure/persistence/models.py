import uuid
from datetime import datetime
from typing import List

from pgvector.sqlalchemy import Vector
from sqlalchemy import ForeignKey, func
from sqlalchemy.orm import Mapped, mapped_column, registry, relationship, declarative_base

table_registry = registry()

Base = declarative_base()

N_DIM = 64


@table_registry.mapped_as_dataclass
class Rating:
    __tablename__ = "ratings"

    id: Mapped[uuid.UUID] = mapped_column(init=False, primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    movie_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("movies.id"))
    rating: Mapped[float]
    timestamp: Mapped[datetime] = mapped_column(server_default=func.now())
    
    user: Mapped["User"] = relationship("User", back_populates="ratings")


@table_registry.mapped_as_dataclass
class User:
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    ratings: Mapped[List[Rating]] = relationship("Rating", back_populates="user", default_factory=list, init=False)

    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())


@table_registry.mapped_as_dataclass
class Movie:
    __tablename__ = "movies"

    id: Mapped[uuid.UUID] = mapped_column(init=False, primary_key=True)
    title: Mapped[str]
    genres: Mapped[str]
    embedding = Vector(N_DIM)
