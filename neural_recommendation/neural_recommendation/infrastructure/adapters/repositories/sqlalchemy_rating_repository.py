from typing import List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.rating import Rating as DomainRating
from neural_recommendation.domain.ports.repositories.rating_repository import RatingRepository
from neural_recommendation.infrastructure.persistence.models import Rating as SQLRating


class SQLAlchemyRatingRepository(RatingRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    async def bulk_create(self, ratings: List[DomainRating]) -> None:
        for rating in ratings:
            sql_rating = SQLRating(user_id=rating.user_id, movie_id=rating.movie_id, rating=rating.rating)
            self.session.add(sql_rating)
        await self.session.commit()

    async def get_by_user_id(self, user_id: int) -> List[DomainRating]:
        sql_ratings = await self.session.scalars(select(SQLRating).where(SQLRating.user_id == user_id))
        rows = sql_ratings.all()
        return [
            DomainRating(
                id=row.id,
                user_id=row.user_id,
                movie_id=row.movie_id,
                timestamp=row.timestamp,
                rating=row.rating,
            )
            for row in rows
        ]
