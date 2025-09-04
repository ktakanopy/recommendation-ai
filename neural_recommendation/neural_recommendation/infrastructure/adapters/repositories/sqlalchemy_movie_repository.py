import uuid
from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.movie import Movie as DomainMovie
from neural_recommendation.domain.ports.repositories.movie_repository import MovieRepository
from neural_recommendation.infrastructure.persistence.models import Movie as SQLMovie


class SQLAlchemyMovieRepository(MovieRepository):
    # TODO: implement the crud in the endpoints
    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, sql_movie: SQLMovie) -> DomainMovie:
        genres_list = sql_movie.genres.split("|") if sql_movie.genres else []
        return DomainMovie(
            id=sql_movie.id, 
            title=sql_movie.title, 
            genres=genres_list
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
        query = select(SQLMovie).where(SQLMovie.id == movie_id)
        result = await self.session.execute(query)
        movie = result.scalar_one_or_none()
        return self._to_domain(movie) if movie else None

    async def get_by_title(self, title: str) -> Optional[DomainMovie]:
        query = select(SQLMovie).where(SQLMovie.title == title)
        result = await self.session.execute(query)
        movie = result.scalar_one_or_none()
        return self._to_domain(movie) if movie else None

    async def get_all(self, offset: int = 0, limit: int = 100) -> List[DomainMovie]:
        query = select(SQLMovie).offset(offset).limit(limit)
        result = await self.session.execute(query)
        movies = result.scalars().all()
        return [self._to_domain(movie) for movie in movies]

    async def create(self, movie: DomainMovie) -> DomainMovie:
        genres_str = "|".join(movie.genres) if movie.genres else ""
        sql_movie = SQLMovie(
            title=movie.title,
            genres=genres_str,
        )
        self.session.add(sql_movie)
        await self.session.commit()
        await self.session.refresh(sql_movie)
        return self._to_domain(sql_movie)

    async def update(self, movie: DomainMovie) -> DomainMovie:
        query = select(SQLMovie).where(SQLMovie.id == movie.id)
        result = await self.session.execute(query)
        sql_movie = result.scalar_one_or_none()
        
        if not sql_movie:
            raise ValueError(f"Movie with id {movie.id} not found")
        
        sql_movie.title = movie.title
        sql_movie.genres = "|".join(movie.genres) if movie.genres else ""
        
        await self.session.commit()
        await self.session.refresh(sql_movie)
        return self._to_domain(sql_movie)

    async def delete(self, movie_id: uuid.UUID) -> bool:
        query = select(SQLMovie).where(SQLMovie.id == movie_id)
        result = await self.session.execute(query)
        movie = result.scalar_one_or_none()
        
        if not movie:
            return False
        
        await self.session.delete(movie)
        await self.session.commit()
        return True
