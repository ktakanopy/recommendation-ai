import uuid
from typing import List, Optional

from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.movie import Movie as DomainMovie
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.infrastructure.persistence.models import Movie as SQLMovie


class SQLAlchemyMovieRepository(MovieRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, sql_movie: SQLMovie) -> DomainMovie:
        return DomainMovie(
            id=sql_movie.id, title=sql_movie.title, genres=sql_movie.genres, embedding=sql_movie.embedding
        )

    def get_similar_movies(
        self, query_embedding: List[float], user_watched_movies: List[uuid.UUID], num_recommendations: int
    ) -> List[DomainMovie]:
        query = (
            self.session.query(SQLMovie, SQLMovie.embedding.cosine_distance(query_embedding))
            .where(SQLMovie.id.notin_(user_watched_movies))
            .order_by(SQLMovie.embedding.cosine_distance(SQLMovie.embedding))
            .limit(num_recommendations)
        )
        result = query.all()
        return [self._to_domain(movie) for movie, _ in result]

    async def get_by_id(self, movie_id: uuid.UUID) -> Optional[DomainMovie]:
        query = self.session.select(SQLMovie).where(SQLMovie.id == movie_id)
        result = await self.session.execute(query)
        movie = result.scalar_one_or_none()
        return self._to_domain(movie) if movie else None

    async def get_by_title(self, title: str) -> Optional[DomainMovie]:
        query = self.session.select(SQLMovie).where(SQLMovie.title == title)
        result = await self.session.execute(query)
        movie = result.scalar_one_or_none()
        return self._to_domain(movie) if movie else None

    async def create(self, movie: DomainMovie) -> DomainMovie:
        sql_movie = SQLMovie(
            id=movie.id,
            title=movie.title,
            genres=movie.genres,
            embedding=movie.embedding,
        )
        self.session.add(sql_movie)
        await self.session.commit()
        return self._to_domain(sql_movie)

    async def update(self, movie: DomainMovie) -> DomainMovie:
        sql_movie = SQLMovie(
            id=movie.id,
            title=movie.title,
            genres=movie.genres,
            embedding=movie.embedding,
        )
        self.session.add(sql_movie)
        await self.session.commit()
        return self._to_domain(sql_movie)
