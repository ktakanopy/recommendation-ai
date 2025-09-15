from typing import List, Optional

from sqlalchemy import delete, select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.rating import Rating
from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.infrastructure.persistence.models import Rating as SQLRating
from neural_recommendation.infrastructure.persistence.models import User as SQLUser


class SQLAlchemyUserRepository(UserRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, sql_user: SQLUser) -> DomainUser:
        return DomainUser(
            username=sql_user.username,
            email=sql_user.email,
            password_hash=sql_user.password,
            age=sql_user.age,
            gender=sql_user.gender,
            occupation=sql_user.occupation,
            id=sql_user.id,
            created_at=sql_user.created_at,
            ratings=None,
        )

    def _to_sql(self, domain_user: DomainUser) -> SQLUser:
        """Convert domain model to SQLAlchemy model"""
        return SQLUser(
            username=domain_user.username,
            email=domain_user.email,
            password=domain_user.password_hash,
            age=domain_user.age,
            gender=domain_user.gender,
            occupation=domain_user.occupation,
        )

    async def create(self, user: DomainUser) -> DomainUser:
        # Step 1: Create the user entity first
        sql_user = SQLUser(
            username=user.username,
            email=user.email,
            password=user.password_hash,
            age=user.age,
            gender=user.gender,
            occupation=user.occupation,
        )
        self.session.add(sql_user)
        await self.session.commit()
        await self.session.refresh(sql_user)

        # Step 2: Create associated ratings if provided
        if user.ratings:
            await self._create_user_ratings(sql_user.id, user.ratings)

        return self._to_domain(sql_user)

    async def _create_user_ratings(self, user_id: int, ratings: List[Rating]) -> None:
        """Private method to create ratings for a user"""
        for rating in ratings:
            sql_rating = SQLRating(
                user_id=user_id, movie_id=rating.movie_id, rating=rating.rating, timestamp=rating.timestamp
            )
            self.session.add(sql_rating)
        await self.session.commit()

    async def get_by_id(self, user_id: int) -> Optional[DomainUser]:
        sql_user = await self.session.scalar(select(SQLUser).where(SQLUser.id == user_id))
        return self._to_domain(sql_user) if sql_user else None

    async def get_by_email(self, email: str) -> Optional[DomainUser]:
        sql_user = await self.session.scalar(select(SQLUser).where(SQLUser.email == email))
        return self._to_domain(sql_user) if sql_user else None

    async def get_by_username(self, username: str) -> Optional[DomainUser]:
        sql_user = await self.session.scalar(select(SQLUser).where(SQLUser.username == username))
        return self._to_domain(sql_user) if sql_user else None

    async def get_by_username_or_email(self, username: str, email: str) -> Optional[DomainUser]:
        sql_user = await self.session.scalar(
            select(SQLUser).where((SQLUser.username == username) | (SQLUser.email == email))
        )
        return self._to_domain(sql_user) if sql_user else None

    async def get_user_ratings(self, user_id: int) -> List[Rating]:
        """Get all ratings for a specific user"""

        sql_ratings = await self.session.scalars(select(SQLRating).where(SQLRating.user_id == user_id))
        ratings = sql_ratings.all()

        domain_ratings = []
        for sql_rating in ratings:
            domain_rating = Rating(
                id=sql_rating.id,
                user_id=sql_rating.user_id,
                movie_id=sql_rating.movie_id,
                rating=sql_rating.rating,
                timestamp=sql_rating.timestamp,
            )
            domain_ratings.append(domain_rating)

        return domain_ratings

    async def get_all(self, offset: int = 0, limit: int = 100) -> List[DomainUser]:
        query = await self.session.scalars(select(SQLUser).offset(offset).limit(limit))
        sql_users = query.all()
        return [self._to_domain(sql_user) for sql_user in sql_users]

    async def update(self, user: DomainUser) -> DomainUser:
        sql_user = await self.session.scalar(select(SQLUser).where(SQLUser.id == user.id))
        if not sql_user:
            raise ValueError("User not found")

        sql_user.username = user.username
        sql_user.email = user.email
        sql_user.password = user.password_hash
        sql_user.age = user.age
        sql_user.gender = user.gender
        sql_user.occupation = user.occupation

        await self.session.commit()
        await self.session.refresh(sql_user)
        return self._to_domain(sql_user)

    async def delete(self, user_id: int) -> bool:
        sql_user = await self.session.scalar(select(SQLUser).where(SQLUser.id == user_id))
        if not sql_user:
            return False

        await self.session.delete(sql_user)
        await self.session.commit()
        return True

    async def delete_all(self) -> int:
        await self.session.execute(delete(SQLRating))
        result_users = await self.session.execute(delete(SQLUser))
        await self.session.commit()
        return result_users.rowcount or 0
