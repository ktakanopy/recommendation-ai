import uuid
from datetime import datetime
from typing import List

from sqlalchemy import ForeignKey, String, func
from sqlalchemy.orm import Mapped, declarative_base, mapped_column, registry, relationship
from sqlalchemy.dialects.postgresql import ARRAY

table_registry = registry()

Base = declarative_base()


@table_registry.mapped_as_dataclass
class Rating:
    __tablename__ = "ratings"

    id: Mapped[uuid.UUID] = mapped_column(init=False, primary_key=True, default=uuid.uuid4)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id"))
    movie_id: Mapped[uuid.UUID] = mapped_column(ForeignKey("movies.id"))
    rating: Mapped[float]
    timestamp: Mapped[datetime] = mapped_column(init=False, server_default=func.now())

    user: Mapped["User"] = relationship("User", back_populates="ratings", init=False)


@table_registry.mapped_as_dataclass
class User:
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(init=False, primary_key=True)
    username: Mapped[str] = mapped_column(unique=True)
    password: Mapped[str]
    email: Mapped[str] = mapped_column(unique=True)
    age: Mapped[int] = mapped_column(nullable=True, default=None)
    gender: Mapped[str] = mapped_column(nullable=True, default=None)
    occupation: Mapped[str] = mapped_column(nullable=True, default=None)
    ratings: Mapped[List[Rating]] = relationship("Rating", back_populates="user", default_factory=list, init=False)

    created_at: Mapped[datetime] = mapped_column(init=False, server_default=func.now())


@table_registry.mapped_as_dataclass
class Movie:
    __tablename__ = "movies"

    id: Mapped[uuid.UUID] = mapped_column(init=False, primary_key=True, default=uuid.uuid4)
    original_id: Mapped[int]
    title: Mapped[str]
    genres: Mapped[List[str]] = mapped_column(ARRAY(String))
