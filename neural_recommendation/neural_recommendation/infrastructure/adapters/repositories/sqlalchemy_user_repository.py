from typing import List, Optional

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from neural_recommendation.domain.models.user import User as DomainUser
from neural_recommendation.domain.ports.repositories.user_repository import UserRepository
from neural_recommendation.infrastructure.persistence.models import User as SQLUser


class SQLAlchemyUserRepository(UserRepository):
    def __init__(self, session: AsyncSession):
        self.session = session

    def _to_domain(self, sql_user: SQLUser) -> DomainUser:
        """Convert SQLAlchemy model to domain model"""
        return DomainUser(
            id=sql_user.id,
            username=sql_user.username,
            email=sql_user.email,
            ratings=sql_user.ratings,
            password_hash=sql_user.password,
            created_at=sql_user.created_at,
        )

    def _to_sql(self, domain_user: DomainUser) -> SQLUser:
        """Convert domain model to SQLAlchemy model"""
        return SQLUser(
            id=domain_user.id,
            username=domain_user.username,
            email=domain_user.email,
            password=domain_user.password_hash,
            ratings=domain_user.ratings,
            created_at=domain_user.created_at,
        )

    async def create(self, user: DomainUser) -> DomainUser:
        sql_user = SQLUser(
            username=user.username,
            email=user.email,
            password=user.password_hash,
            ratings=user.ratings,
        )
        self.session.add(sql_user)
        await self.session.commit()
        await self.session.refresh(sql_user)
        return self._to_domain(sql_user)

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
